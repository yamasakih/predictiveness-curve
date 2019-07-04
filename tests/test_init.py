import pkg_resources

import pytest

import predictiveness_curve


def test_version() -> None:
    expect = pkg_resources.get_distribution('predictiveness_curve').version
    actual = predictiveness_curve.__version__
    assert expect == actual


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
