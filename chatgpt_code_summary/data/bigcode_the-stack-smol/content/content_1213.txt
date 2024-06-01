from probs import Binomial


class TestBinomial:
    @staticmethod
    def test_binomial() -> None:
        d = Binomial()

        assert d.expectation() == 0
        assert d.variance() == 0
        # TODO: Python 3.7 implementation differs from 3.8+
        # assert P(d == 0) == 1
        # assert P(d == 1) == 0
        # assert P(d == 2) == 0

        # d = Binomial(n=6, p=0.7)

        # assert P(d == 0) == 0.000729
        # assert P(d == 1) == 0.010206
        # assert P(d == 2) == 0.059535
        # assert P(d == 3) == 0.18522
        # assert P(d == 4) == 0.324135
        # assert P(d == 5) == 0.302526
        # assert P(d == 6) == 0.117649
        # assert P(d == 7) == 0

    @staticmethod
    def test_sum() -> None:
        d = Binomial() + Binomial()

        assert d.expectation() == 0
        assert d.variance() == 0
        # TODO
        assert d.pmf == {}
        # assert P(d == 2) == 1 / 36
        # assert P(d == 8) == 5 / 36
        # assert P(d == 60) == 0

    @staticmethod
    def test_repr() -> None:
        d = Binomial() + Binomial()

        assert str(d) == "Binomial(pmf={}, n=0, p=1)"
