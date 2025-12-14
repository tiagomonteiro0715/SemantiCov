from unittest import TestCase, main
from semantic_coverage_tool import calculation_total, compare, game_play


class game_tests(TestCase):
    def test_calculation_total_sum(self):
        self.assertEqual(calculation_total([5, 4, 3]), 12)

    # def test_calculation_total_first_case(self):
    #     self.assertEqual(calculation_total([11, 10]), 0)

    # def test_calculation_total_second_case(self):
    #     self.assertEqual(calculation_total([3, 5, 7, 11]), 16)

    def test_compare_function(self):
        # self.assertEqual(compare(12, 12), "It's a draw")
        # self.assertEqual(compare(0, 13), "You win with a BlackJack")
        self.assertEqual(compare(19, 0), "Your opponent wins with a BlackJack")
        self.assertEqual(compare(22, 17), "You went over. You lose!")
        self.assertEqual(compare(18, 25), "You win! Opponent went over.")
        self.assertEqual(compare(20, 17), "You win")
        self.assertEqual(compare(16, 19), "You lose")

    def test_compare_type_error(self):
        with self.assertRaises(TypeError):
            compare("Dsa", 2)

    def test_calculation_total_error_case_one(self):
        with self.assertRaises(TypeError):
            calculation_total("21321")

    def test_calculation_total_error_case_two(self):
        with self.assertRaises(TypeError):
            calculation_total([12, 3, "Hello", 2])

    def test_game_play(self):
        inputs = iter(['y', 'n'])
        game_play(lambda _: next(inputs))


if __name__ == '__main__':
    main()
