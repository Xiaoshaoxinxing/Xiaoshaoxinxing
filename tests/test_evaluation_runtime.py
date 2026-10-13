import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from mer_omni.evaluation import ContextCheckedLLM


class ContextCapacityTests(unittest.TestCase):
    def test_preserves_prompt_and_sampling_for_valid_request(self):
        backend = Mock()
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        checked = ContextCheckedLLM(backend, tokenizer, 8)
        prompts = ['unchanged prompt']
        sampling = SimpleNamespace(max_tokens=5, temperature=0.7)
        self.assertIs(checked.generate(prompts, sampling), backend.generate.return_value)
        backend.generate.assert_called_once_with(prompts, sampling)

    def test_rejects_overlong_prompt_before_generation(self):
        backend = Mock()
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4]
        checked = ContextCheckedLLM(backend, tokenizer, 8)
        with self.assertRaisesRegex(ValueError, 'refusing to truncate'):
            checked.generate(['long prompt'], SimpleNamespace(max_tokens=5))
        backend.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
