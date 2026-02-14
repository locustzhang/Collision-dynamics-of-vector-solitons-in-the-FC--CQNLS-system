import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import time
import random
import warnings
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")


# ===================== 1. 全局配置 & 顶刊风格 =====================
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "Paper_Sweep_Trends_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[Info] Device: {DEVICE}")
print(f"[Info] Output Directory: {OUTPUT_DIR}")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.linewidth": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "legend.frameon": False,
    "legend.fontsize": 11
})


# ===================== 2. 物理核心 =====================
class CQNLS_System:
    def __init__(self, N=2048, L=100.0, alpha=1.8, coeffs=None):
        self.N = N
        self.L = L
        self.dx = 2 * L / N
        self.alpha = alpha
        self.coeffs = coeffs if coeffs else {'a': 1.0, 'g': 1.0, 'b': -0.5}

        self.x = torch.linspace(-L, L, N, dtype=torch.float64, device=DEVICE)
        self.k = (2 * np.pi / (2 * L)) * torch.fft.fftfreq(N).to(DEVICE, dtype=torch.float64) * N
        self.k_alpha = torch.abs(self.k) ** alpha

    def get_hamiltonian(self, psi):
        """返回粒子数 (norm) 作为主要守恒诊断"""
        rho_tot = (psi.abs() ** 2).sum(dim=0)
        norm = rho_tot.sum().item() * self.dx
        return norm

    def find_ground_state(self, max_iter=8000, dtau=0.001):
        psi = torch.exp(-self.x ** 2 / 5.0) + 0j
        a, b = self.coeffs['a'], self.coeffs['b']
        prev_norm = 0.0

        print(f"  [基态求解] α={self.alpha:.2f} 开始 (max_iter={max_iter}, dtau={dtau})")

        for it in range(max_iter):
            psi = torch.fft.ifft(torch.fft.fft(psi) * torch.exp(-self.k_alpha * dtau))
            rho = psi.abs() ** 2
            psi *= torch.exp((a * rho + b * rho ** 2) * dtau)
            current_P = torch.sum(psi.abs() ** 2).item() * self.dx
            psi *= np.sqrt(5.0 / max(1e-12, current_P))

            if it % 500 == 0:
                norm_diff = abs(current_P - prev_norm) / max(1e-12, current_P)
                print(f"    iter {it:5d}/{max_iter} | norm={current_P:8.6f} | Δnorm={norm_diff:.2e}")
                if norm_diff < 1e-6 and it > 2000:
                    print("      → 收敛达成，提前结束")
                    break
                prev_norm = current_P

        print("  [基态] 完成")
        return psi

    def _find_soliton_centers(self, rho_np, x_np, initial_v):
        rho_max = rho_np.max()
        char_width = np.sum(rho_np > 0.1 * rho_max) * self.dx / 2
        distance = int(np.clip(char_width * 0.8, 15, 50))
        prom_levels = [0.1, 0.05, 0.03, 0.02]

        peaks = []
        for prom in prom_levels:
            peaks, _ = find_peaks(rho_np, prominence=prom * rho_max, distance=distance)
            if len(peaks) >= 2:
                break

        if len(peaks) >= 2:
            top_idx = np.argsort(rho_np[peaks])[-2:]
            top_peaks = peaks[top_idx]
            pos = x_np[top_peaks]
            sort_idx = np.argsort(pos)
            return pos[sort_idx[0]], pos[sort_idx[1]]

        elif len(peaks) == 1:
            center = x_np[peaks[0]]
            half_max = 0.5 * rho_np[peaks[0]]
            left_idx = np.where(rho_np[:peaks[0]] < half_max)[0]
            right_idx = np.where(rho_np[peaks[0]:] < half_max)[0]
            if len(left_idx) > 0 and len(right_idx) > 0:
                fwhm = x_np[peaks[0] + right_idx[0]] - x_np[left_idx[-1]]
                sep = fwhm * 0.3
            else:
                sep = 2.0
            return center - sep, center + sep

        else:
            mid = len(x_np) // 2
            left = np.sum(x_np[:mid] * rho_np[:mid]) / (np.sum(rho_np[:mid]) + 1e-12)
            right = np.sum(x_np[mid:] * rho_np[mid:]) / (np.sum(rho_np[mid:]) + 1e-12)
            return left, right

    def run_collision_metrics(self, psi_init, T_max=50.0, dt=0.0005, initial_v=0.5):
        psi = psi_init.clone()
        steps = int(T_max / dt)
        save_int = max(1, steps // 200)  # ~200 个记录点

        rho_init = (psi.abs() ** 2).sum(dim=0)
        max_rho_0 = rho_init.max().item()
        total_particles_0 = rho_init.sum().item() * self.dx
        energy_0 = self.get_hamiltonian(psi)

        cm_history_L = []
        cm_history_R = []
        time_points = []
        max_rho_history = []
        total_particles_history = []
        energy_history = []

        x_np = self.x.cpu().numpy()

        print(f"  [RTE 开始] T_max={T_max:.1f}, dt={dt:.6f}, steps={steps:,}, initial_v={initial_v:.3f}")

        start_time = time.time()
        for i in range(steps + 1):
            # SSFM 演化
            lin = torch.exp(-0.5j * self.k_alpha * dt)
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * lin, dim=1)
            p1, p2 = psi[0], psi[1]
            r1, r2 = p1.abs() ** 2, p2.abs() ** 2
            V1 = self.coeffs['a'] * r1 + self.coeffs['g'] * r2 + self.coeffs['b'] * r1 ** 2
            V2 = self.coeffs['g'] * r1 + self.coeffs['a'] * r2 + self.coeffs['b'] * r2 ** 2
            p1 *= torch.exp(1j * V1 * dt)
            p2 *= torch.exp(1j * V2 * dt)
            psi = torch.stack([p1, p2])
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * lin, dim=1)

            # 详细诊断
            if i % save_int == 0 or i == steps:
                t_now = i * dt
                rho_tot = (psi.abs() ** 2).sum(dim=0)
                rho_np = rho_tot.cpu().numpy()
                current_norm = rho_tot.sum().item() * self.dx
                current_max_rho = rho_tot.max().item()

                cm_L, cm_R = self._find_soliton_centers(rho_np, x_np, initial_v)
                min_sep = abs(cm_R - cm_L)

                norm_err = (current_norm / max(1e-12, total_particles_0) - 1) * 100
                rad_est = (1.0 - current_max_rho / max(1e-12, max_rho_0)) * 100

                progress = (i / steps) * 100
                elapsed = time.time() - start_time
                eta = elapsed / max(1e-6, progress) * (100 - progress) if progress > 0 else 0

                print(f"  t={t_now:6.2f} ({progress:5.1f}%) | "
                      f"Norm Err={norm_err:+8.5f}% | "
                      f"ρ_max={current_max_rho:8.4f} | Rad Est={rad_est:6.2f}% | "
                      f"Sep={min_sep:6.3f} | CM=({cm_L:6.3f}, {cm_R:6.3f}) | "
                      f"ETA ~{eta/60:.1f} min")

                # 异常预警（红色风格模拟，用大写 WARNING）
                if abs(norm_err) > 0.001:
                    print(f"  *** WARNING *** Norm 漂移异常: {norm_err:+.5f}%")
                if min_sep < 1.0:
                    print(f"  *** 注意 *** 孤子分离过小: {min_sep:.3f} (可能合并)")

                cm_history_L.append(cm_L)
                cm_history_R.append(cm_R)
                time_points.append(t_now)
                max_rho_history.append(current_max_rho)
                total_particles_history.append(current_norm)
                energy_history.append(self.get_hamiltonian(psi))

        print(f"  [RTE 完成] 总耗时 {time.time() - start_time:.1f} 秒")

        # 后处理（保持原逻辑）
        cm_L = np.array(cm_history_L)
        cm_R = np.array(cm_history_R)
        t_arr = np.array(time_points)
        max_rho_arr = np.array(max_rho_history)
        particles_arr = np.array(total_particles_history)
        energy_arr = np.array(energy_history)

        dist = np.abs(cm_R - cm_L)
        min_dist = np.min(dist)
        collision_time = t_arr[np.argmin(dist)] if len(dist) > 0 else 0.0

        radiation = (1.0 - max_rho_arr[-1] / max(1e-12, max_rho_0)) * 100

        skip = 3
        pre_len = max(int(len(t_arr) * 0.1), skip + 2)
        post_len = max(int(len(t_arr) * 0.2), 5)

        if pre_len > skip + 2:
            v_init_L = np.polyfit(t_arr[skip:pre_len], cm_L[skip:pre_len], 1)[0]
            v_init_R = np.polyfit(t_arr[skip:pre_len], cm_R[skip:pre_len], 1)[0]
        else:
            v_init_L = -initial_v
            v_init_R = initial_v

        if post_len > 2:
            v_final_L = np.polyfit(t_arr[-post_len:], cm_L[-post_len:], 1)[0]
            v_final_R = np.polyfit(t_arr[-post_len:], cm_R[-post_len:], 1)[0]
        else:
            v_final_L = v_init_L
            v_final_R = v_init_R

        compression = np.max(max_rho_arr) / max(1e-12, max_rho_0)

        delta_v_L = np.abs(v_final_L - v_init_L)
        delta_v_R = np.abs(v_final_R - v_init_R)
        avg_v_init = (np.abs(v_init_L) + np.abs(v_init_R)) / 2
        vel_change = ((delta_v_L + delta_v_R) / 2 / avg_v_init) * 100 if avg_v_init > 1e-6 else 0.0

        momentum_error = np.abs((v_final_L + v_final_R) - (v_init_L + v_init_R))

        final_norm_error = (particles_arr[-1] / max(1e-12, total_particles_0) - 1) * 100
        final_energy_error = (energy_arr[-1] / max(1e-12, energy_0) - 1) * 100

        bound_threshold = 2.0
        is_bound = (min_dist < bound_threshold) and \
                   (abs(v_final_L) < 0.1 * abs(initial_v)) and \
                   (abs(v_final_R) < 0.1 * abs(initial_v))

        return {
            "Radiation (%)": radiation,
            "Min Separation": min_dist,
            "Compression Ratio": compression,
            "Max Density Final": max_rho_arr[-1],
            "Velocity Change (%)": vel_change,
            "Final Velocity L": v_final_L,
            "Final Velocity R": v_final_R,
            "Initial Velocity L": v_init_L,
            "Initial Velocity R": v_init_R,
            "Momentum Error": momentum_error,
            "Particle Loss (%)": (1.0 - particles_arr[-1] / max(1e-12, total_particles_0)) * 100,
            "Final Norm Error (%)": final_norm_error,
            "Final Energy Error (%)": final_energy_error,
            "Initial Energy": energy_0,
            "Final Energy": energy_arr[-1],
            "Collision Time": collision_time,
            "Is Bound State": is_bound,
            "param": 0.0
        }


# ===================== 3. 扫描逻辑引擎 =====================
def run_full_sweeps():
    print("\n=== 开始全参数扫描 (最终平衡版) ===\n")
    print(f"[参数] N=2048, L=100.0, dt=0.0005")
    print(f"[说明] 实时详细打印，便于监控数值稳定性\n")

    # Velocity Sweep
    print(">>> 1. Velocity Sweep...")
    velocities = np.concatenate([
        np.linspace(0.2, 0.3, 3),
        np.linspace(0.3, 0.5, 5),
        np.linspace(0.5, 0.9, 5)
    ])
    print(f"    扫描 {len(velocities)} 个速度点: {velocities[0]:.2f} → {velocities[-1]:.2f}")

    res_v = []
    sys_base = CQNLS_System(alpha=1.8, N=2048, L=100.0)
    phi = sys_base.find_ground_state()
    phi_np = phi.cpu().numpy()
    x_np = sys_base.x.cpu().numpy()

    for idx, v in enumerate(velocities, 1):
        print(f"\n→ 模拟 {idx}/{len(velocities)} | v = {v:.2f}")
        shift_idx = int(20.0 / sys_base.dx)
        T_dynamic = max(40.0, 40.0 / max(0.1, v))

        p_L = np.roll(phi_np, -shift_idx) * np.exp(1j * v * x_np)
        p_R = np.roll(phi_np, shift_idx) * np.exp(-1j * v * x_np)

        norm_L = np.sum(np.abs(p_L) ** 2) * sys_base.dx
        norm_R = np.sum(np.abs(p_R) ** 2) * sys_base.dx

        if norm_L < 0.1:
            print(f"  [Warning] Low norm_L = {norm_L:.6f}")

        p_L *= np.sqrt(2.5 / max(0.1, norm_L))
        p_R *= np.sqrt(2.5 / max(0.1, norm_R))

        psi = torch.tensor(np.stack([p_L, p_R]), device=DEVICE, dtype=torch.complex128)

        metrics = sys_base.run_collision_metrics(psi, T_max=T_dynamic, dt=0.0005, initial_v=v)
        metrics["param"] = v
        res_v.append(metrics)

        print(f"  [总结] v={v:.2f} → Rad={metrics['Radiation (%)']:6.2f}%, "
              f"Comp={metrics['Compression Ratio']:.2f}, VelΔ={metrics['Velocity Change (%)']:5.2f}%, "
              f"MinSep={metrics['Min Separation']:5.2f}, Norm Err={metrics['Final Norm Error (%)']:6.4f}%, "
              f"Bound={metrics['Is Bound State']}")
        print("-" * 80)

    df_v = pd.DataFrame(res_v)
    plot_six_subplots(df_v, "Velocity", "Initial Velocity $v$")

    # Alpha Sweep
    print("\n>>> 2. Alpha Sweep...")
    alphas = np.concatenate([np.linspace(1.2, 1.95, 9), [2.0]])
    print(f"    扫描 {len(alphas)} 个 α 值: {alphas[0]:.2f} → {alphas[-1]:.2f}")

    res_a = []
    v_fix = 0.5
    T_dynamic_alpha = 50.0

    for idx, alpha in enumerate(alphas, 1):
        print(f"\n→ 模拟 {idx}/{len(alphas)} | α = {alpha:.2f}")
        sys = CQNLS_System(alpha=alpha, N=2048, L=100.0)
        phi = sys.find_ground_state()
        phi_np = phi.cpu().numpy()
        x_np = sys.x.cpu().numpy()

        shift_idx = int(20.0 / sys.dx)
        p_L = np.roll(phi_np, -shift_idx) * np.exp(1j * v_fix * x_np)
        p_R = np.roll(phi_np, shift_idx) * np.exp(-1j * v_fix * x_np)

        norm_L = np.sum(np.abs(p_L) ** 2) * sys.dx
        norm_R = np.sum(np.abs(p_R) ** 2) * sys.dx

        if norm_L < 0.1:
            print(f"  [Warning] Low norm_L = {norm_L:.6f}")

        p_L *= np.sqrt(2.5 / max(0.1, norm_L))
        p_R *= np.sqrt(2.5 / max(0.1, norm_R))

        psi = torch.tensor(np.stack([p_L, p_R]), device=DEVICE, dtype=torch.complex128)

        metrics = sys.run_collision_metrics(psi, T_max=T_dynamic_alpha, dt=0.0005, initial_v=v_fix)
        metrics["param"] = alpha
        res_a.append(metrics)

        print(f"  [总结] α={alpha:.2f} → Rad={metrics['Radiation (%)']:6.2f}%, "
              f"Comp={metrics['Compression Ratio']:.2f}, VelΔ={metrics['Velocity Change (%)']:5.2f}%, "
              f"MinSep={metrics['Min Separation']:5.2f}, Norm Err={metrics['Final Norm Error (%)']:6.4f}%, "
              f"Bound={metrics['Is Bound State']}")
        print("-" * 80)

    df_a = pd.DataFrame(res_a)
    plot_six_subplots(df_a, "Alpha", r"Lévy Index $\alpha$")


# ===================== 4. 顶刊风格六子图 =====================
def plot_six_subplots(df, name, xlabel):
    print(f"  [Plotting] Generating six-subplot figure for {name}...")

    df.to_csv(os.path.join(OUTPUT_DIR, f"Data_{name}.csv"), index=False)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)

    c_rad = "#D62728"
    c_cmp = "#1F77B4"
    c_sep = "#2CA02C"
    c_den = "#FF7F0E"
    c_vel = "#9467BD"
    c_norm = "#17BECF"

    x = df["param"]
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    ax1 = fig.add_subplot(gs[0, 0])
    ln1 = ax1.plot(x, df["Radiation (%)"], 'o-', color=c_rad, label="Radiation",
                   markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(x, df["Radiation (%)"], color=c_rad, alpha=0.1)
    ax1.set_xlabel(xlabel, fontweight='bold')
    ax1.set_ylabel("Radiation Loss (%)", color=c_rad, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=c_rad)
    ax1_twin = ax1.twinx()
    ln2 = ax1_twin.plot(x, df["Compression Ratio"], 's--', color=c_cmp, label="Compression",
                        markerfacecolor='white', markeredgewidth=2)
    ax1_twin.set_ylabel("Compression Ratio", color=c_cmp, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=c_cmp)
    ax1.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], loc='best')
    ax1.set_title(f"({subplot_labels[0]}) Inelasticity & Deformation", loc='left', fontsize=14)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, df["Min Separation"], 'D-', color=c_sep, lw=2,
             markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(x, df["Min Separation"], color=c_sep, alpha=0.1)
    ax2.set_xlabel(xlabel, fontweight='bold')
    ax2.set_ylabel("Min Separation (a.u.)", fontweight='bold')
    ax2.set_title(f"({subplot_labels[1]}) Interaction Range", loc='left', fontsize=14)
    ax2.set_ylim(bottom=0)

    merger_limit = 4.0
    min_val = df["Min Separation"].min()
    print(f"  [Debug] Min separation across sweep: {min_val:.4f} (Threshold: {merger_limit})")

    if min_val < merger_limit:
        ax2.axhspan(0, merger_limit, color='gray', alpha=0.2, label='Strong Interaction Zone')
        text_str = "Merger / Collapse" if min_val < 1.0 else "Strong Overlap"
        ax2.text(x.iloc[len(x)//2], merger_limit/2, text_str, ha='center', va='center',
                 color='dimgray', fontsize=10, fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax2.legend(loc='best')

    ax2.set_axisbelow(True)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, df["Max Density Final"], '^-', color=c_den, lw=2,
             markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(x, df["Max Density Final"], color=c_den, alpha=0.1)
    ax3.set_xlabel(xlabel, fontweight='bold')
    ax3.set_ylabel("Final Max Density (a.u.)", fontweight='bold')
    ax3.set_title(f"({subplot_labels[2]}) Post-Collision Density", loc='left', fontsize=14)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(x, df["Velocity Change (%)"], 'p-', color=c_vel, lw=2,
             markerfacecolor='white', markeredgewidth=2)
    ax4.fill_between(x, df["Velocity Change (%)"], color=c_vel, alpha=0.1)
    ax4.set_xlabel(xlabel, fontweight='bold')
    ax4.set_ylabel("Velocity Change (%)", fontweight='bold')
    ax4.set_title(f"({subplot_labels[3]}) Collision-Induced Velocity Shift", loc='left', fontsize=14)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(x, df["Final Velocity L"], 'h-', color=c_vel, lw=2,
             markerfacecolor='white', markeredgewidth=2)
    ax5.fill_between(x, df["Final Velocity L"], color=c_vel, alpha=0.1)
    ax5.set_xlabel(xlabel, fontweight='bold')
    ax5.set_ylabel("Final Velocity (Left Soliton)", fontweight='bold')
    ax5.set_title(f"({subplot_labels[4]}) Left Soliton Final Velocity", loc='left', fontsize=14)

    ax6 = fig.add_subplot(gs[1, 2])
    norm_err = (df["Final Energy"] - df["Initial Energy"]) / np.abs(df["Initial Energy"]).clip(lower=1e-12) * 100
    ax6.semilogy(x, np.abs(norm_err).clip(lower=1e-8), '*-', color=c_norm, lw=2.5,
                 markerfacecolor='white', markeredgewidth=2)
    ax6.fill_between(x, 1e-8, np.abs(norm_err).clip(lower=1e-8), color=c_norm, alpha=0.12)
    ax6.set_xlabel(xlabel, fontweight='bold')
    ax6.set_ylabel("Norm Conservation Error (%)", fontweight='bold')
    ax6.set_title(f"({subplot_labels[5]}) Particle Number Conservation", loc='left', fontsize=14)
    ax6.axhline(0.01, color='gray', ls='--', alpha=0.7, label='0.01% threshold')
    ax6.legend(loc='best')
    ax6.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"SixSubplots_{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved SixSubplots_{name}.png (300 DPI)")


# ===================== 主程序 =====================
if __name__ == "__main__":
    t0 = time.time()
    run_full_sweeps()
    total_time = time.time() - t0
    print(f"\n[Done] Total runtime: {total_time:.2f}s ({total_time / 60:.1f} mins)")
    print(f"All results saved in: {os.path.abspath(OUTPUT_DIR)}")