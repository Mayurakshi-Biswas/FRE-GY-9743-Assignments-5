"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        # 1) Which curve discounts this cashflow? Read it from the funding parameter.
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        # 2) Cache the payment date and the fixed cash amount C = N * alpha.
        #    Notional is signed (negative = we pay), so C carries the sign too.
        self.payment_date_ = product.payment_date
        self.cash_amount_ = product.notional * product.accrued

    def calculate_value(self) -> None:
        if self.payment_date_ < self.value_date:
            # Already paid before today: nothing left to value, no cash today.
            self.value_ = 0.0
            self.cash_ = 0.0
        elif self.payment_date_ == self.value_date:
            # Paid today: worth exactly C (DF = 1), and it shows up as cash today.
            self.value_ = self.cash_amount_
            self.cash_ = self.cash_amount_
        else:
            # Future payment: PV = C * D_funding(T_p), no cash today.
            df = self.model.discount_factor(self.funding_index_, self.payment_date_)
            self.value_ = self.cash_amount_ * df
            self.cash_ = 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        # Make sure there is one block per model component, with the right sizes.
        self.model.resize_gradient(gradient)

        if self.payment_date_ > self.value_date:
            # dPV/dstate = C * dD_funding/dstate. Fold C into the scaler and let the
            # model write (or add) the chained DF gradient into every block.
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.payment_date_, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate,
            )
        elif not accumulate:
            # Past or same-day payment: PV doesn't depend on the curve -> zero risk.
            # Overwrite mode must still clear any old values; accumulate adds nothing.
            for block in gradient:
                block.fill(0.0)


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)