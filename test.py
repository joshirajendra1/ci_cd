import pandas as pd

def test_greater():
    num = 100

    assert num > 100


def test_greater_equal():
    num = 100
    assert num >= 100


def test_less():
    num = 100
    assert num < 200


test_less()

test_greater_equal()

test_greater()
