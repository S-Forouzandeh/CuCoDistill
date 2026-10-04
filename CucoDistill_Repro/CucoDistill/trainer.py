"""CuCoTrainer: correct curriculum co-evolutionary training (teacher not frozen)."""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from hypergraph import Hypergraph
from model import CuCoModel
from aked import AKED
from curriculum import SpectralCurriculum
from losses import Losses
from theory import Theory
from bench import Bench


class CuCoTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.model = CuCoModel(cfg).to(self.dev)
        self.aked = AKED(cfg).to(self.dev)
        self.curr = SpectralCurriculum(cfg)
        # a SINGLE optimiser over the whole model -> teacher & student co-evolve
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.loss_hist: List[float] = []
        self.task_curve: List[float] = []
        self.K = 1

    @staticmethod
    def _acc(logits, labels, mask) -> float:
        return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()

    def _pack(self, X, hg):
        return (X, hg, hg.spectral_operator(), hg.comembership_mask(), hg.degree_features())

    # ---- Phase 1: warm up the teacher (and shared backbone) ----
    def pretrain_teacher(self, X, hg, labels, masks, verbose=True):
        pack = self._pack(X, hg)
        for ep in range(self.cfg.pretrain_epochs):
            self.model.train()
            self.opt.zero_grad()
            t = self.model.teacher_forward(*pack)
            loss = F.cross_entropy(t["logits"][masks["train"]], labels[masks["train"]])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            if verbose and (ep + 1) % 10 == 0:
                print(f"  [pretrain] ep {ep+1:3d}  loss {loss.item():.4f}  "
                      f"train_acc {self._acc(t['logits'], labels, masks['train']):.3f}")

    # ---- Phase 2: curriculum co-evolutionary distillation ----
    def distill(self, X, hg, labels, masks, verbose=True):
        pack = self._pack(X, hg)
        spec, A, degf = pack[2], pack[3], pack[4]
        self.K = math.ceil(self.cfg.topk_alpha * hg.max_node_edges())
        hist = {"student_acc": [], "teacher_acc": [], "val_acc": [], "loss": []}
        best = 0.0

        for ep in range(self.cfg.epochs):
            self.model.train()
            self.opt.zero_grad()

            # forward both paths on the clean view (teacher is NOT frozen) -----
            t = self.model.teacher_forward(X, hg, spec, A, degf)
            s = self.model.student_forward(X, hg, spec, A, degf, self.K)

            # AKED augmented view, then student forward on it -----------------
            X_aug, H_aug = self.aked(X, hg.H, t["attn"].detach(), t["emb"].detach(),
                                     s["emb"].detach(), ep, self.cfg.epochs, training=True)
            hg_aug = Hypergraph(H_aug)
            s_aug = self.model.student_forward(
                X_aug, hg_aug, hg_aug.spectral_operator(),
                hg_aug.comembership_mask(), hg_aug.degree_features(), self.K)

            # curriculum difficulties + gates ---------------------------------
            d_con = self.curr.contrastive_difficulty(s["emb"], s_aug["emb"])
            d_dis = self.curr.distill_difficulty(t["emb"], s["emb"])
            g_dis = self.curr.gate(d_dis, ep, "distill")        # (N,)
            g_con = self.curr.gate(d_con, ep, "contrast")
            lam1, lam2, lam3 = self.curr.loss_weights(ep)

            # losses ----------------------------------------------------------
            L_task = (F.cross_entropy(t["logits"][masks["train"]], labels[masks["train"]])
                      + F.cross_entropy(s["logits"][masks["train"]], labels[masks["train"]]))
            L_embed = Losses.embed_align(s["aligned"], t["emb"], g_dis)
            L_attn = Losses.attn_transfer(t["attn"], s["attn"], A.float())
            L_feat = Losses.feat_match(s["feats"], t["feats"], self.cfg.feat_layer_gamma)
            L_kd = Losses.soft_kd(s["logits"], t["logits"], self.cfg.kd_temperature)
            L_distill = L_embed + L_attn + L_feat + L_kd
            nce = Losses.info_nce(s["emb"], s_aug["emb"], self.cfg.infonce_tau)
            L_contrast = (g_con * nce).sum() / (g_con.sum() + 1e-9)

            loss = lam1 * L_distill + lam2 * L_contrast + lam3 * L_task
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.loss_hist.append(loss.item())
            with torch.no_grad():
                self.task_curve.append(F.cross_entropy(
                    s["logits"][masks["train"]], labels[masks["train"]]).item())

            with torch.no_grad():
                s_acc = self._acc(s["logits"], labels, masks["train"])
                t_acc = self._acc(t["logits"], labels, masks["train"])
                v_acc = self._acc(s["logits"], labels, masks["val"])
            best = max(best, v_acc)
            for k, v in zip(("student_acc", "teacher_acc", "val_acc", "loss"),
                            (s_acc, t_acc, v_acc, loss.item())):
                hist[k].append(v)
            if verbose and (ep + 1) % 15 == 0:
                print(f"  [distill] ep {ep+1:3d}  L {loss.item():.3f}  "
                      f"(l1 {lam1:.2f}/l2 {lam2:.2f})  "
                      f"s_acc {s_acc:.3f}  t_acc {t_acc:.3f}  val {v_acc:.3f}")
        return hist

    # ---- Phase 3: evaluation, theorems, efficiency ----
    def evaluate(self, X, hg, labels, masks):
        pack = self._pack(X, hg)
        self.model.eval()
        with torch.no_grad():
            t = self.model.teacher_forward(*pack)
            s = self.model.student_forward(*pack, self.K)
        t_acc = self._acc(t["logits"], labels, masks["test"])
        s_acc = self._acc(s["logits"], labels, masks["test"])
        d_eff = hg.effective_dimension(self.cfg.deff_threshold)

        res = {
            "teacher_test_acc": t_acc,
            "student_test_acc": s_acc,
            "gain_pp": (s_acc - t_acc) * 100,
            "theorem1": Theory.t1_spectral(t["attn"], s["attn"], hg.N,
                                           hg.max_node_edges(), self.cfg.spectral_eps),
            "theorem2": Theory.t2_convergence(self.task_curve),
            "theorem3": Theory.t3_generalisation(
                int(masks["train"].sum()),
                Bench.params(self.model.student_layers) + Bench.params(self.model.student_cls)),
            "theorem4_diagnostic": Theory.t4_diagnostic(X, self.K, d_eff),
            "efficiency": Bench.report(self.model, pack, self.K),
        }
        self._print(res)
        return res

    @staticmethod
    def _print(r):
        e, t4 = r["efficiency"], r["theorem4_diagnostic"]
        print("\n" + "=" * 64)
        print("  CuCoDistill - results")
        print("=" * 64)
        print(f"  Teacher test acc : {r['teacher_test_acc']*100:6.2f}%")
        print(f"  Student test acc : {r['student_test_acc']*100:6.2f}%   "
              f"(gain {r['gain_pp']:+.2f} pp)")
        t1 = r['theorem1']
        print(f"  Theorem 1 [spectral]  ||A_s-A_t||_F {t1['frob_error']:.3f} <= "
              f"bound {t1['paper_bound']:.3f}  -> {t1['satisfied']}  "
              f"(implied eps {t1['implied_eps']:.3f})")
        print(f"  Theorem 2 [converge]  log-log slope {r['theorem2']['slope']:.3f} "
              f"(target {r['theorem2']['target']})")
        print(f"  Theorem 3 [general.]  bound {r['theorem3']['bound']:.4f}")
        print(f"  Theorem 4 [R(X)>R*]   R(X) {t4['R_x']:.2f}  R* {t4['R_star']:.2f}  "
              f"K {t4['K']}  d_eff {t4['d_eff']}  -> predict student-superior: "
              f"{t4['predict_student_superior']}")
        print(f"  Efficiency:  measured speedup {e['measured_speedup']:.1f}x  "
              f"| theoretical |V|/K = {e['theoretical_speedup_NoverK']:.1f}x  "
              f"| param ratio {e['param_ratio']:.2f}x")
        print("  " + e["note"])
        print("=" * 64)
