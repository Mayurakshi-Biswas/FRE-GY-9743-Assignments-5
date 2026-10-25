"""IFR curve objects and interpolator setup based on the course template."""

import numpy as np
import QuantLib as ql
from typing import List, Optional

from fixedincomelib.date import Date, accrued
from fixedincomelib.data import DataCollection
from fixedincomelib.model import (
    Model, ModelComponent, ModelType, BuildMethodCollection,
    ModelBuilderRegistry, ModelDeserializerRegistry,
)
from fixedincomelib.product import Product
from fixedincomelib.utilities import Interpolator1D, InterpolatorFactory
from fixedincomelib.yield_curve.build_method import YieldCurveIndexBuildMethod, YieldCurveFundingBuildMethod

__all__ = ["YieldCurve", "YieldCurveModelComponent"]


class YieldCurve(Model):

    _version = 1
    _model_type = ModelType.YIELD_CURVE

    def __init__(
        self,
        value_date: Date,
        data_collection: DataCollection,
        build_method_collection: BuildMethodCollection,
    ) -> None:
        super().__init__(value_date, self._model_type, data_collection, build_method_collection)

    def serialize(self) -> dict:
        content = {}
        content["VERSION"] = YieldCurve._version
        content["MODEL_TYPE"] = YieldCurve._model_type.to_string()
        content["VALUE_DATE"] = self.value_date.ISO()
        content["BUILD_METHOD_COLLECTION"] = self.build_method_collection.serialize()
        content["DATA_COLLECTION"] = self.data_collection.serialize()
        return content

    @classmethod
    def deserialize(cls, input_dict: dict) -> "YieldCurve":
        input_dict_ = input_dict.copy()
        assert "VERSION" in input_dict_
        version = input_dict_["VERSION"]
        assert "MODEL_TYPE" in input_dict_
        model_type = input_dict_["MODEL_TYPE"]
        assert "VALUE_DATE" in input_dict_
        value_date = Date(input_dict_["VALUE_DATE"])
        bmc = BuildMethodCollection.deserialize(input_dict_["BUILD_METHOD_COLLECTION"])
        dc = DataCollection.deserialize(input_dict_["DATA_COLLECTION"])
        # find modelbuilder
        func = ModelBuilderRegistry().get(model_type)
        return func(value_date, dc, bmc)

    def discount_factor(
        self, index: ql.Index | str, expiry_date: Date, payment_currency: Optional[str] = None
    ):
        """Combine local discount factors recursively through component references.

        ``payment_currency`` is reserved for future use and does not affect the
        calculation. The reference is read from each stored component identifier.
        """
        visited = set()

        def discount_factor_with_reference(component_index):
            this_component: YieldCurveModelComponent = self.retrieve_model_component(
                component_index
            )
            this_index = this_component.component_identifier
            component_name = this_index.name()
            if component_name in visited:
                raise ValueError(f"Circular yield-curve reference: {component_name}")
            visited.add(component_name)

            df = 1.0
            reference_index = getattr(this_index, "reference_index", None)
            if reference_index is not None:
                df = discount_factor_with_reference(reference_index)
            return df * this_component.discount_factor(expiry_date)

        return discount_factor_with_reference(index)

    def discount_factor_gradient_wrt_state(
        self, index: ql.Index | str, expiry_date: Date,
        gradient_vector: List[np.ndarray],
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write raw DF derivatives into component-ordered blocks in place."""
        # Total DF along a chain:  D = D_local * D_reference  (e.g. D_funding = D_spread * D_SOFR)
        # Product rule:
        #   dD/d(local states)     = D_reference * dD_local/d(local states)
        #   dD/d(reference states) = D_local     * dD_reference/d(reference states)
        # The reference may itself have a reference, so we recurse, carrying a running
        # multiplier = product of the DFs of every component already visited above it.

        # Allocate one zero block per component (in component order) if needed.
        self.resize_gradient(gradient_vector)
        # Overwrite mode: clear the old values once, then add every contribution below.
        if not accumulate:
            for block in gradient_vector:
                block.fill(0.0)

        visited = set()

        def add_gradient_with_reference(component_index, multiplier: float) -> float:
            """Add this component's (and its references') DF gradient; return its full DF."""
            this_component: YieldCurveModelComponent = self.retrieve_model_component(
                component_index
            )
            this_index = this_component.component_identifier
            component_name = this_index.name()
            if component_name in visited:
                raise ValueError(f"Circular yield-curve reference: {component_name}")
            visited.add(component_name)

            local_df = this_component.discount_factor(expiry_date)

            # 1) Reference curve first: its states are scaled by this component's local DF.
            reference_df = 1.0
            reference_index = getattr(this_index, "reference_index", None)
            if reference_index is not None:
                reference_df = add_gradient_with_reference(
                    reference_index, multiplier * local_df
                )

            # 2) This component's own states: scaled by the reference chain's DF.
            block = gradient_vector[self.retrieve_model_component_index(component_name)]
            this_component.discount_factor_gradient_wrt_state(
                expiry_date, block, scaler=multiplier * reference_df, accumulate=True
            )
            return reference_df * local_df

        add_gradient_with_reference(index, scaler)


class YieldCurveModelComponent(ModelComponent):

    def __init__(
        self,
        value_date: Date,
        component_identifier: ql.Index,
        state_data: np.ndarray,
        build_method: YieldCurveIndexBuildMethod | YieldCurveFundingBuildMethod,
        calibration_product: Optional[List[Product]] = None,
        calibration_funding: Optional[List[str]] = None,
        market_data: Optional[List] = None,
    ) -> None:

        super().__init__(
            value_date,
            component_identifier,
            state_data,
            build_method,
            [] if calibration_product is None else calibration_product,
            [] if calibration_funding is None else calibration_funding,
            [] if market_data is None else market_data,
        )
        assert len(state_data) == 2
        self.num_state_data_ = len(state_data[0])
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            state_data[0],
            state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    def perturb_model_parameter(
        self, parameter_id: int, perturb_size: float, override_parameter: Optional[bool] = False
    ):
        super().perturb_model_parameter(parameter_id, perturb_size, override_parameter)
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            self.state_data[0],
            self.state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    @property
    def state_data_interpolator(self) -> Interpolator1D:
        return self.interpolator_

    @property
    def num_state_data(self) -> int:
        return self.num_state_data_

    def discount_factor(self, expiry_date: Date):
        """Discount using this component's IFR only, excluding any reference curve."""
        if expiry_date.serialNumber() == 0:
            raise ValueError("A valid discount-factor date is required.")
        if expiry_date < self.value_date:
            raise ValueError("Discount-factor date cannot precede the model value date.")

        # Use the same time convention as the IFR node construction.
        time_to_expiry = accrued(self.value_date, expiry_date)
        exponent = self.state_data_interpolator.integrate(0.0, time_to_expiry)
        return np.exp(-exponent)

    def discount_factor_gradient_wrt_state(
        self, expiry_date: Date, gradient_vector: np.ndarray,
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write raw local DF derivatives into the supplied array in place."""
        # D(T) = exp(-I(T)),  I(T) = integral_0^T f(t) dt  (f = this component's IFR)
        # dD/df_k = -D(T) * dI/df_k   (chain rule through the exponential)
        df = self.discount_factor(expiry_date)  # also validates expiry_date
        time_to_expiry = accrued(self.value_date, expiry_date)  # same convention as discount_factor
        d_integral = self.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(
            0.0, time_to_expiry
        )
        contribution = scaler * (-df) * np.asarray(d_integral, dtype=float)

        if gradient_vector.shape != contribution.shape:
            raise ValueError(
                f"gradient_vector has shape {gradient_vector.shape}, expected {contribution.shape}"
            )
        if accumulate:
            gradient_vector += contribution      # add to whatever is already there
        else:
            gradient_vector[:] = contribution    # overwrite in place (keep the same array object)


ModelDeserializerRegistry().register(YieldCurve._model_type.to_string(), YieldCurve.deserialize)