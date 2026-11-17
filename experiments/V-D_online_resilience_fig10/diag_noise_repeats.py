
import os, sys, json, ast, time
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import sequence_breaker_noise as sbn
sbn.BASE_URL = sys.argv[1]; out = sys.argv[2]; mode = sys.argv[3]; n = int(sys.argv[4]); breaker_closed = sys.argv[5] == "closed"
conds = [("noise_off", False, 0.0, 0.0), ("noise_v_and_s", True, 0.005, 0.00001), ("noise_v_only", True, 0.005, 0.0), ("noise_s_only", True, 0.0, 0.00001)]
rows = []
sbn.del_mods(); sbn.set_breaker(breaker_closed); sbn.set_router(sbn.ROUTER_MODE[mode])
for rep in range(n):
    for name, en, vn, sn in conds:
        sbn.set_noise(enabled=en, v_noise_std=vn, s_noise_std=sn); time.sleep(0.2)
        df = sbn.run_sequence(f"{mode}_{name}_{rep}", rate=20.0)
        def cq(m):
            d = m if isinstance(m, dict) else {}
            return 1e3 * sum(float(v.get("dQ_mvar", 0)) for k, v in d.items() if str(k) != "61")
        q = df["mods"].apply(cq); t = df["i"] / 20.0
        dist = q[(t >= 1.0) & (t <= 3.0)]; steady = q[(t >= 2.5) & (t <= 2.95)]
        act = dist[dist > 5.0]
        rows.append(dict(cond=name, rep=rep, q_steady=steady.mean(), q_peak=dist.max(), q_std=dist[t[dist.index] >= 1.3].std(),
                         n_zero_dropouts=int(((q < 5.0) & (t >= 1.4) & (t <= 2.95)).sum()),
                         t_first_action=float(t[act.index[0]]) if len(act) else None,
                         vmax_true_like=float(df["max_voltage"][(t >= 2.5) & (t <= 2.95)].mean())))
sbn.set_noise(enabled=False, v_noise_std=0.0, s_noise_std=0.0); sbn.del_mods(); sbn.set_router("none", settle_s=0.0); sbn.set_breaker(True)
pd.DataFrame(rows).to_csv(out, index=False)
