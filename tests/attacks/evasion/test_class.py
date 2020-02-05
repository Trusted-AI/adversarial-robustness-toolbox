
# Run with pytest -q tests/attacks/evasion/test_class.py
# or  python -m pytest  tests/attacks/evasion/test_class.py
# or python -m pytest  tests/attacks/evasion
# or pytest -q tests/attacks/evasion/test_class.py --cmdopt=type2

import pytest


class TestClass:
    def test_thatPasses(self):
        x = "this"
        assert "h" in x

    def test_thatFails(self):
        x = "hello"
        assert x == "hello", "This test was supposed to do that {0}".format(1)

    def test_answer(cmdopt):
        if cmdopt == "type1":
            print("first")
        elif cmdopt == "type2":
            print("second")
        # assert 0  # to see what was printed
