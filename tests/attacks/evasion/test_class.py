
# Run with pytest -q tests/attacks/evasion/test_class.py
# or  python -m pytest  tests/attacks/evasion/test_class.py

class TestClass:
    def test_thatPasses(self):
        x = "this"
        assert "h" in x

    def test_thatFails(self):
        x = "hello"
        assert x == "hello", "This test was supposed to do that {0}".format(1)
