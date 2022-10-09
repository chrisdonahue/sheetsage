import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    CTCTransducer,
    IdentityDecoder,
    IdentityEncoder,
    MLPEncoder,
    S2STransducer,
    TransformerDecoder,
    TransformerEncoder,
    _PositionalEmbedding,
)


class TestModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.src_max_len = 50
        self.src_dim = 100
        self.tgt_max_len = 10
        self.tgt_vocab_size = 2000
        with torch.no_grad():
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            self.src = torch.randn(
                self.src_max_len, self.batch_size, self.src_dim
            ).float()
            self.src_len = torch.randint(
                low=1, high=self.src_max_len, size=(self.batch_size,), dtype=torch.long
            )
            self.tgt = torch.randint(
                low=0,
                high=self.tgt_vocab_size,
                size=(self.tgt_max_len, self.batch_size),
                dtype=torch.long,
            )
            self.tgt_len = torch.randint(
                low=1, high=self.tgt_max_len, size=(self.batch_size,), dtype=torch.long
            )
            self.tgt_len = np.minimum(self.tgt_len, self.src_len)

    def test_identity_encoder(self):
        with torch.no_grad():
            enc = IdentityEncoder(self.src_dim)
            self.assertEqual(enc.get_src_enc_dim(), self.src_dim)
            params = [(n, list(p.shape)) for n, p in enc.named_parameters()]
            self.assertEqual(len(params), 0)

            enc.train()
            out = enc(self.src, self.src_len)
            self.assertTrue(np.array_equal(out.cpu().numpy(), self.src.cpu().numpy()))
            self.assertAlmostEqual(torch.abs(out).sum().item(), 19973.3, places=1)

    def test_mlp_encoder(self):
        with torch.no_grad():
            enc = MLPEncoder(self.src_dim)
            self.assertEqual(enc.get_src_enc_dim(), 512)
            params = [(n, list(p.shape)) for n, p in enc.named_parameters()]
            self.assertEqual(len(params), 2)
            self.assertEqual(params[0][0], "hidden_0.weight")
            self.assertEqual(params[0][1], [512, self.src_dim])
            self.assertEqual(params[1][0], "hidden_0.bias")
            self.assertEqual(params[1][1], [512])

            for train in [False, True]:
                if train:
                    enc.train()
                else:
                    enc.eval()
                out = enc(self.src, self.src_len)
                self.assertEqual(
                    list(out.shape), [self.src_max_len, self.batch_size, 512]
                )
                out_hat = enc(self.src, self.src_len)
                self.assertEqual(
                    list(out_hat.shape), [self.src_max_len, self.batch_size, 512]
                )
                if train:
                    self.assertFalse(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                else:
                    self.assertTrue(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                    self.assertAlmostEqual(
                        torch.abs(out).sum().item(), 29487.5, places=1
                    )

    def test_transformer_encoder(self):
        class LegacyTransformerEncoder(TransformerEncoder):
            def __init__(self, input_dim, model_dim=512):
                nn.Module.__init__(self)
                self.emb = nn.Linear(input_dim, model_dim)
                self.pos_emb = _PositionalEmbedding(model_dim)
                self.dropout = nn.Dropout(p=0.1)
                self.src_emb_dim = model_dim
                super().__init__(
                    model_dim,
                    model_dim=model_dim,
                    _legacy_for_unit_test=True,
                )

            def forward(self, src_emb, src_len):
                src_max_len, batch_size, _ = src_emb.shape
                x = src_emb
                x = x.view(src_max_len * batch_size, -1)
                x = self.emb(x)
                x = x.view(src_max_len, batch_size, -1)
                x = self.pos_emb(x)
                x = self.dropout(x)
                return super().forward(x, src_len)

        with torch.no_grad():
            enc = LegacyTransformerEncoder(self.src_dim)
            self.assertEqual(enc.get_src_enc_dim(), 512)
            params = [(n, list(p.shape)) for n, p in enc.named_parameters()]
            self.assertEqual(len(params), 76)
            self.assertEqual(params[0][0], "emb.weight")
            self.assertEqual(params[0][1], [512, self.src_dim])
            self.assertEqual(params[1][0], "emb.bias")
            self.assertEqual(params[1][1], [512])

            # Check values
            for train in [False, True]:
                if train:
                    enc.train()
                else:
                    enc.eval()
                out = enc(self.src, self.src_len)
                self.assertEqual(
                    list(out.shape), [self.src_max_len, self.batch_size, 512]
                )
                out_hat = enc(self.src, self.src_len)
                self.assertEqual(
                    list(out_hat.shape), [self.src_max_len, self.batch_size, 512]
                )
                if train:
                    self.assertFalse(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                else:
                    self.assertTrue(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                    self.assertAlmostEqual(
                        torch.abs(out).sum().item(), 101794.5, places=1
                    )

            # Create dummy input w/ different padding
            src = self.src
            seq_idxs = torch.arange(0, self.src_max_len, dtype=torch.long).expand(
                self.batch_size, -1
            )
            mask = seq_idxs < self.src_len.unsqueeze(1)
            mask = mask.transpose(0, 1).unsqueeze(2)
            src_hat = torch.where(mask, src, torch.randn_like(src))
            self.assertTrue(np.array_equal(src[0, 0].numpy(), src_hat[0, 0].numpy()))
            self.assertFalse(np.array_equal(src[-1, 0].numpy(), src_hat[-1, 0].numpy()))
            self.assertFalse(np.array_equal(src.numpy(), src_hat.numpy()))

            # Ensure result is same (i.e., that src_len padding works)
            enc.eval()
            out = enc(src, self.src_len)
            out_hat = enc(src_hat, self.src_len)
            self.assertFalse(np.array_equal(out, out_hat))
            for i in range(self.batch_size):
                a = out[: self.src_len[i], i].cpu().numpy()
                b = out_hat[: self.src_len[i], i].cpu().numpy()
                self.assertTrue(np.array_equal(a, b))

            # Test skip embedding
            enc = TransformerEncoder(16, model_dim=16)
            self.assertEqual(len(list(enc.named_parameters())), 74)
            with self.assertRaises(ValueError):
                TransformerEncoder(17, model_dim=16)

    def test_transformer_decoder(self):
        class EmbeddingTransformerDecoder(TransformerDecoder):
            def __init__(self, input_dim, vocab_size, model_dim=512):
                super().__init__(model_dim, model_dim, model_dim=model_dim)
                self.src_emb = nn.Linear(input_dim, model_dim)
                self.tgt_emb = nn.Embedding(vocab_size, model_dim)

            def forward(self, src, src_len, tgt, tgt_len):
                src_max_len, batch_size, _ = src.shape
                src_enc = src
                src_enc = src_enc.view(src_max_len * batch_size, -1)
                src_enc = self.src_emb(src_enc)
                src_enc = src_enc.view(src_max_len, batch_size, -1)
                tgt_emb = self.tgt_emb(tgt)
                return super().forward(src_enc, src_len, tgt_emb, tgt_len)

        with torch.no_grad():
            dec = EmbeddingTransformerDecoder(self.src_dim, self.tgt_vocab_size)
            self.assertEqual(dec.get_tgt_dec_dim(), 512)
            params = [(n, list(p.shape)) for n, p in dec.named_parameters()]
            self.assertEqual(len(params), 113)
            self.assertEqual(params[-3][0], "src_emb.weight")
            self.assertEqual(params[-3][1], [512, self.src_dim])
            self.assertEqual(params[-2][0], "src_emb.bias")
            self.assertEqual(params[-2][1], [512])
            self.assertEqual(params[-1][0], "tgt_emb.weight")
            self.assertEqual(params[-1][1], [2000, 512])

            # Check values
            for train in [False, True]:
                if train:
                    dec.train()
                else:
                    dec.eval()
                out = dec(self.src, self.src_len, self.tgt, self.tgt_len)
                self.assertEqual(
                    list(out.shape), [self.tgt_max_len, self.batch_size, 512]
                )
                out_hat = dec(self.src, self.src_len, self.tgt, self.tgt_len)
                self.assertEqual(
                    list(out_hat.shape), [self.tgt_max_len, self.batch_size, 512]
                )
                if train:
                    self.assertFalse(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                else:
                    self.assertTrue(
                        np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy())
                    )
                    self.assertAlmostEqual(
                        torch.abs(out).sum().item(), 20384.9, places=1
                    )

            # Create dummy src input w/ different padding
            src = self.src
            seq_idxs = torch.arange(0, self.src_max_len, dtype=torch.long).expand(
                self.batch_size, -1
            )
            mask = seq_idxs < self.src_len.unsqueeze(1)
            mask = mask.transpose(0, 1).unsqueeze(2)
            src_hat = torch.where(mask, src, torch.randn_like(src))
            self.assertTrue(np.array_equal(src[0, 0].numpy(), src_hat[0, 0].numpy()))
            self.assertFalse(np.array_equal(src[-1, 0].numpy(), src_hat[-1, 0].numpy()))
            self.assertFalse(np.array_equal(src.numpy(), src_hat.numpy()))

            # Create dummy tgt input w/ different padding
            tgt = self.tgt
            seq_idxs = torch.arange(0, self.tgt_max_len, dtype=torch.long).expand(
                self.batch_size, -1
            )
            mask = seq_idxs < self.tgt_len.unsqueeze(1)
            mask = mask.transpose(0, 1)
            tgt_hat = torch.where(
                mask,
                tgt,
                torch.randint(
                    low=0,
                    high=self.tgt_vocab_size,
                    size=(self.tgt_max_len, self.batch_size),
                    dtype=torch.long,
                ),
            )

            # Ensure result is same (i.e., that masking works)
            dec.eval()
            out = dec(src, self.src_len, tgt, self.tgt_len)
            out_hat = dec(src_hat, self.src_len, tgt_hat, self.tgt_len)
            self.assertFalse(np.array_equal(out, out_hat))
            for i in range(self.batch_size):
                a = out[: self.tgt_len[i], i].cpu().numpy()
                b = out_hat[: self.tgt_len[i], i].cpu().numpy()
                self.assertTrue(np.array_equal(a, b))

            # Test skip embedding
            enc = TransformerEncoder(16, model_dim=16)
            self.assertEqual(len(list(enc.named_parameters())), 74)
            with self.assertRaises(ValueError):
                TransformerEncoder(17, model_dim=16)

    def test_ctc_transducer(self):
        with torch.no_grad():
            model = CTCTransducer(self.tgt_vocab_size, src_dim=self.src_dim)
            params = [(n, list(p.shape)) for n, p in model.named_parameters()]
            self.assertEqual(len(params), 2)
            self.assertEqual(params[0][0], "output.weight")
            self.assertEqual(params[0][1], [self.tgt_vocab_size + 1, self.src_dim])
            self.assertEqual(params[1][0], "output.bias")
            self.assertEqual(params[1][1], [self.tgt_vocab_size + 1])

            model.train()
            out = model(self.src, self.src_len)
            self.assertEqual(
                list(out.shape),
                [self.src_max_len, self.batch_size, self.tgt_vocab_size + 1],
            )
            out_hat = model(self.src, self.src_len)
            self.assertEqual(
                list(out_hat.shape),
                [self.src_max_len, self.batch_size, self.tgt_vocab_size + 1],
            )
            self.assertTrue(np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy()))
            self.assertAlmostEqual(torch.abs(out).sum().item(), 231966.6, places=1)

            loss = F.ctc_loss(
                F.log_softmax(out, dim=-1),
                self.tgt.permute(1, 0),
                self.src_len,
                self.tgt_len,
                blank=self.tgt_vocab_size,
            )
            self.assertAlmostEqual(loss.item(), 29.0, places=1)

            with self.assertRaisesRegex(ValueError, "Invalid sequence lengths"):
                model(self.src.permute(1, 0, 2), self.src_len)

    def test_s2s_transducer(self):
        with torch.no_grad():
            model = S2STransducer(
                self.tgt_vocab_size,
                src_dim=self.src_dim,
                tgt_emb_mode="embed",
                tgt_vocab_size=self.tgt_vocab_size,
                tgt_emb_dim=self.src_dim,
                dec_cls=IdentityDecoder,
            )
            params = [(n, list(p.shape)) for n, p in model.named_parameters()]
            self.assertEqual(len(params), 3)
            self.assertEqual(params[0][0], "tgt_emb.embedding.weight")
            self.assertEqual(params[0][1], [self.tgt_vocab_size, self.src_dim])
            self.assertEqual(params[1][0], "output.weight")
            self.assertEqual(params[1][1], [self.tgt_vocab_size, self.src_dim])
            self.assertEqual(params[2][0], "output.bias")
            self.assertEqual(params[2][1], [self.tgt_vocab_size])

            model.train()
            out = model(self.src, self.src_len, self.tgt, self.tgt_len)
            self.assertEqual(
                list(out.shape),
                [self.tgt_max_len, self.batch_size, self.tgt_vocab_size],
            )
            out_hat = model(self.src, self.src_len, self.tgt, self.tgt_len)
            self.assertEqual(
                list(out_hat.shape),
                [self.tgt_max_len, self.batch_size, self.tgt_vocab_size],
            )
            self.assertTrue(np.array_equal(out.cpu().numpy(), out_hat.cpu().numpy()))
            self.assertAlmostEqual(torch.abs(out).sum().item(), 462012.9, places=1)

            loss = F.cross_entropy(
                out[:-1].view((self.tgt_max_len - 1) * self.batch_size, -1),
                self.tgt[1:].view((self.tgt_max_len - 1) * self.batch_size),
                ignore_index=-1,
                reduction="mean",
            )

            self.assertAlmostEqual(loss.item(), 22.7, places=1)
