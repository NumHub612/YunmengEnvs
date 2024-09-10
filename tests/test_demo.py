# -*- encoding: utf-8 -*-
import unittest


class TestDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_demo(self):
        return True


if __name__ == "__main__":
    # 在项目根目录下测试
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        # 逐次加载测试
        suit = unittest.TestSuite()
        suit.addTest(TestDemo("test_demo"))

        # 执行测试
        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
