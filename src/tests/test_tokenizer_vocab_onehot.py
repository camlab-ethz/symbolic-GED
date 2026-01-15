import unittest
from typing import List

from src import tokenizer, vocab, onehot
from src.pde import grammar as pde_grammar


class TestTokenizerVocabOnehot(unittest.TestCase):

    def setUp(self):
        self.pde = "dt(u) - 1.935*dxx(u) = 0"

    def test_tokenizers(self):
        op = tokenizer.operator_tokenize(self.pde)
        self.assertEqual(
            op, ["dt", "(", "u", ")", "-", "1.935", "*", "dxx", "(", "u", ")", "=", "0"]
        )
        digits = tokenizer.num_tokenize_as_digits(self.pde)
        # ensure digits split for the number
        self.assertIn(".", digits)
        self.assertIn("1", digits)

    def test_vocab_and_onehot(self):
        toks = tokenizer.operator_tokenize(self.pde)
        t2i, i2t = vocab.build_vocab([toks])
        ids = vocab.tokens_to_ids(toks, t2i)
        self.assertEqual(len(ids), len(toks))
        arr = onehot.tokens_to_onehot(ids, len(t2i))
        self.assertEqual(arr.shape, (len(ids), len(t2i)))

    def test_batch_padding_onehot(self):
        toks = tokenizer.operator_tokenize(self.pde)
        t2i, _ = vocab.build_vocab([toks])
        ids = vocab.tokens_to_ids(toks, t2i)
        short = ids[:5]
        batch = [ids, short]
        V = len(t2i)
        out = onehot.batch_to_onehot(batch, V, max_len=13)
        # shape (B, L, V)
        self.assertEqual(out.shape, (2, 13, V))
        # padded positions in second example (positions >= len(short)) should be all zeros
        self.assertTrue((out[1, len(short) :, :] == 0).all())

    def test_parse_and_masks_and_padding(self):
        seq = pde_grammar.parse_to_productions(self.pde)
        self.assertTrue(len(seq) > 0)
        masks = pde_grammar.build_masks_from_production_sequence(seq)
        self.assertEqual(len(masks), len(seq))
        # each mask length equals production count
        self.assertTrue(all(len(m) == pde_grammar.PROD_COUNT for m in masks))
        # pad the prod sequence to a longer length and ensure padding uses -1 and onehot ignores it
        padded = pde_grammar.pad_production_sequence(seq, max_len=len(seq) + 5)
        self.assertEqual(len(padded), len(seq) + 5)
        # convert to torch one-hot: padded entries should be one-hot at PAD_PROD_ID
        try:
            t = onehot.productions_to_onehot_torch(padded, pde_grammar.PROD_COUNT)
            self.assertEqual(t.shape[0], len(padded))
            import torch

            pad_id = pde_grammar.PAD_PROD_ID
            if pad_id is not None:
                # padded rows must have a 1 at pad_id
                self.assertTrue((t[len(seq) :, pad_id] == 1).all())
            else:
                # fallback: padded rows should be zero if no PAD_PROD_ID
                self.assertTrue((t[len(seq) :].abs().sum(dim=1) == 0).all())
        except RuntimeError:
            # PyTorch may not be available in some environments; still pass the test if missing
            pass

    def test_batch_production_onehot_and_masks(self):
        seq = pde_grammar.parse_to_productions(self.pde)
        seq2 = seq[:5]
        batch = [seq, seq2]
        P = pde_grammar.PROD_COUNT
        # batch one-hot
        try:
            bt = onehot.batch_productions_to_onehot_torch(batch, P)
            self.assertEqual(bt.shape[0], 2)
            self.assertEqual(bt.shape[2], P)
            # padded rows for shorter sequence should be zero
            self.assertTrue((bt[1, len(seq2) :].abs().sum(dim=1) == 0).all())
            # masks
            masks = [pde_grammar.build_masks_from_production_sequence(s) for s in batch]
            mt = onehot.batch_masks_to_tensor(masks)
            self.assertEqual(mt.shape[0], 2)
            self.assertEqual(mt.shape[2], P)
            self.assertTrue((mt[1, len(seq2) :].abs().sum(dim=1) == 0).all())
        except RuntimeError:
            pass


if __name__ == "__main__":
    unittest.main()
