
# Run with pytest -q tests/attacks/evasion/test_class.py
# or  python -m pytest  tests/attacks/evasion/test_class.py
# or python -m pytest  tests/attacks/evasion
# or pytest -q tests/attacks/evasion/test_class.py --cmdopt=type2

import pytest


def test_thatPasses():
    print("Hello1")
    x = "this"
    assert "h" in x


def test_thatFails():
    print("Hello2")
    x = "hello"
    assert x == "hello", "This test was supposed to do that {0}".format(1)


def test_answer(cmdopt):
    print("Hello3")
    if cmdopt == "type1":
        print("first")
    elif cmdopt == "type2":
        print("second")
    # assert 0  # to see what was printed

# class TestClass:
#     def test_thatPasses(self):
#         print("Hello1")
#         x = "this"
#         assert "h" in x
#
#     def test_thatFails(self):
#         print("Hello2")
#         x = "hello"
#         assert x == "hello", "This test was supposed to do that {0}".format(1)
#
#     def test_answer(cmdopt):
#         print("Hello3")
#         if cmdopt == "type1":
#             print("first")
#         elif cmdopt == "type2":
#             print("second")
#         # assert 0  # to see what was printed
