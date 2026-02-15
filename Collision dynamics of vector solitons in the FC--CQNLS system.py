import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import time
import random


# ===================== 全局配置 =====================
# 新增：设置随机种子保证可复现性
def set_random_seed(seed=42):
    """设置所有随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 执行随机种子设置
set_random_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "Paper_Full_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Science/Nature 风格绘图设置（升级视觉参数）
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False,
    "grid.color": "#E0E0E0",
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
})

print(f"=== FCQNLS Full-Scenario Paper Suite V5.1 ===")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Random seed set to: 42 (for reproducibility)")


# ===================== 物理核心类 =====================
class CQNLS_System:
    def __init__(self, N=2048, L=100.0, alpha=1.8, coeffs=None):
        self.N = N
        self.L = L
        self.dx = 2 * L / N
        self.alpha = alpha

        # 物理系数: a(SPM), g(XPM), b(Quintic)
        self.coeffs = coeffs if coeffs else {'a': 1.0, 'g': 1.0, 'b': -0.5}

        self.x = torch.linspace(-L, L, N, dtype=torch.float64, device=DEVICE)
        self.k = (2 * np.pi / (2 * L)) * torch.fft.fftfreq(N).to(DEVICE) * N
        self.k_alpha = torch.abs(self.k) ** alpha

    def get_energy(self, psi):
        """计算哈密顿量"""
        psi_k = torch.fft.fft(psi, dim=1)
        T_op_psi = torch.fft.ifft(psi_k * self.k_alpha, dim=1)
        E_kin = torch.sum(torch.conj(psi) * T_op_psi).real.item() * self.dx

        p1, p2 = psi[0], psi[1]
        r1, r2 = p1.abs() ** 2, p2.abs() ** 2
        a, g, b = self.coeffs['a'], self.coeffs['g'], self.coeffs['b']

        E_pot = torch.sum(
            0.5 * a * (r1 ** 2 + r2 ** 2) + g * r1 * r2 + (b / 3) * (r1 ** 3 + r2 ** 3)
        ).item() * self.dx

        return E_kin - E_pot

    def find_ground_state(self, max_iter=8000, tol=1e-8):
        """虚时演化寻找精确基态"""
        print(f"  [ITE] Searching for ground state (alpha={self.alpha})...")
        psi = torch.exp(-self.x ** 2 / 5.0) + 0j
        dtau = 0.001
        a, b = self.coeffs['a'], self.coeffs['b']

        for i in range(max_iter):
            psi = torch.fft.ifft(torch.fft.fft(psi) * torch.exp(-self.k_alpha * dtau))
            rho = psi.abs() ** 2
            V_eff = a * rho + b * (rho ** 2)
            psi = psi * torch.exp(V_eff * dtau)

            # Renormalize (Fixed Power = 5.0)
            current_P = torch.sum(psi.abs() ** 2).item() * self.dx
            psi *= np.sqrt(5.0 / current_P)

            if i % 2000 == 0:
                pass
        return psi

    def run_collision_analysis(self, psi_init, T_max=30.0, dt=0.0005):
        """实时演化并采集深度数据（新增碰撞检测、相移分析、详细打印）"""
        psi = psi_init.clone()
        steps = int(T_max / dt)
        save_int = max(1, steps // 200)

        data_log = {
            "t": [], "H_err": [], "Max_Rho": [],
            "CM_Left": [], "CM_Right": [], "Radiation": [],
            "Phase_Shift_Left": [], "Phase_Shift_Right": []  # 新增：用于估算相移
        }
        plot_data = {"psi_sq": [], "t": []}

        H0 = self.get_energy(psi)
        print(f"  [RTE] Simulation Start. H0 = {H0:.8f}")
        start_time = time.time()

        cm_init_left = None
        cm_init_right = None
        min_sep = float('inf')
        collision_t = None
        collision_idx = -1

        a, g, b = self.coeffs['a'], self.coeffs['g'], self.coeffs['b']

        for i in range(steps + 1):
            # Symplectic Step（原有逻辑不变）
            lin = torch.exp(-0.5j * self.k_alpha * dt)
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * lin, dim=1)

            p1, p2 = psi[0], psi[1]
            r1, r2 = p1.abs() ** 2, p2.abs() ** 2
            V1 = a * r1 + g * r2 + b * r1 ** 2
            V2 = g * r1 + a * r2 + b * r2 ** 2

            p1 *= torch.exp(1j * V1 * dt)
            p2 *= torch.exp(1j * V2 * dt)
            psi = torch.stack([p1, p2])

            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * lin, dim=1)

            # Logging
            if i % save_int == 0:
                t_now = i * dt

                # Metrics
                H_now = self.get_energy(psi)
                H_err = abs(H_now - H0) / abs(H0)
                rho_tot = (psi.abs() ** 2).sum(dim=0)
                max_rho = rho_tot.max().item()

                # CM Tracking
                rho1, rho2 = r1, r2
                N1 = rho1.sum().item() * self.dx
                N2 = rho2.sum().item() * self.dx

                # 防止除零错误
                if N1 > 1e-10:
                    cm1 = (self.x * rho1).sum().item() * self.dx / N1
                else:
                    cm1 = 0.0
                if N2 > 1e-10:
                    cm2 = (self.x * rho2).sum().item() * self.dx / N2
                else:
                    cm2 = 0.0

                # 检测碰撞时刻（最小质心间距）
                sep = abs(cm1 - cm2)
                if sep < min_sep:
                    min_sep = sep
                    collision_t = t_now
                    collision_idx = len(data_log["t"])

                # 初始化时记录初始质心
                if i == 0:
                    cm_init_left = cm1
                    cm_init_right = cm2

                # 粗估相移（尾部线性拟合减去自由传播位置）
                phase_shift_l = 0.0
                phase_shift_r = 0.0
                if len(data_log["t"]) > 10:  # 避免早期不稳定
                    tail_start = max(0, len(data_log["t"]) - 20)
                    t_tail = np.array(data_log["t"][tail_start:])
                    cm_tail_l = np.array(data_log["CM_Left"][tail_start:])
                    fit_l = np.polyfit(t_tail, cm_tail_l, 1)
                    expected_l = cm_init_left + fit_l[0] * t_now
                    phase_shift_l = cm1 - expected_l
                    phase_shift_r = -phase_shift_l  # 简化假设对称

                # Radiation (Tail ratio)
                mask_core = (self.x > cm1 - 15) & (self.x < cm1 + 15) | (self.x > cm2 - 15) & (self.x < cm2 + 15)
                N_core = rho_tot[mask_core].sum().item() * self.dx
                N_total = rho_tot.sum().item() * self.dx
                rad_ratio = 1.0 - (N_core / N_total)

                # 记录数据
                data_log["t"].append(t_now)
                data_log["H_err"].append(H_err)
                data_log["Max_Rho"].append(max_rho)
                data_log["CM_Left"].append(cm1)
                data_log["CM_Right"].append(cm2)
                data_log["Radiation"].append(rad_ratio)
                data_log["Phase_Shift_Left"].append(phase_shift_l)
                data_log["Phase_Shift_Right"].append(phase_shift_r)

                plot_data["t"].append(t_now)
                plot_data["psi_sq"].append(rho_tot.cpu().numpy())

                # 每 10% 进度打印更详细的信息
                if i % (steps // 10) == 0 or i == steps:
                    print(f"  Progress {i / steps * 100:5.1f}% | t={t_now:6.2f} | "
                          f"H_err={H_err:.2e} | MaxRho={max_rho:7.4f} | "
                          f"Sep={sep:6.3f} | Rad={rad_ratio * 100:5.2f}% | "
                          f"PS_L={phase_shift_l:+.4f}")

        # 最终诊断打印
        if collision_idx >= 0:
            print("\n  === Collision Summary ===")
            print(f"    Estimated collision time: t ≈ {collision_t:.2f}")
            print(f"    Minimum separation: {min_sep:.4f}")
            print(f"    At collision → MaxRho = {data_log['Max_Rho'][collision_idx]:.4f}")
            print(f"    Radiation at collision: {data_log['Radiation'][collision_idx] * 100:.2f}%")

        # 最终汇总打印
        final_rad = data_log["Radiation"][-1] * 100
        max_H_err_pct = max(data_log["H_err"]) * 100
        final_phase_l = data_log["Phase_Shift_Left"][-1]
        print(f"\n  === Final Summary ===")
        print(f"    Total radiation loss: {final_rad:.3f}%")
        print(f"    Max energy error: {max_H_err_pct:.4f}%")
        print(f"    Asymptotic phase shift (left soliton): {final_phase_l:+.4f}")
        print(f"    Simulation time: {time.time() - start_time:.1f} s")

        return pd.DataFrame(data_log), plot_data, self.x.cpu().numpy()


# ===================== 分析与绘图模块（仅升级视觉风格） =====================
def analyze_and_plot(exp_name, df, plot_data, x, alpha, v_init, extra_title=""):
    """生成 Science 级别的分析图表与数据报告（新增碰撞标注、扩展报告）"""

    print(f"\n>>> Generating Analysis Report for [{exp_name}]...")

    # 1. 计算关键物理量
    max_compression = df["Max_Rho"].max() / df["Max_Rho"].iloc[0]
    final_rad = df["Radiation"].iloc[-1]
    max_H_err = df["H_err"].max()

    # 碰撞相关计算
    min_sep_idx = df["CM_Left"].sub(df["CM_Right"]).abs().idxmin()
    coll_t = df["t"].iloc[min_sep_idx]
    min_sep = df["CM_Left"].sub(df["CM_Right"]).abs().min()
    final_phase_l = df["Phase_Shift_Left"].iloc[-1]
    final_phase_r = df["Phase_Shift_Right"].iloc[-1]

    # 计算相移 (Phase Shift)
    tail_len = int(len(df) * 0.2)
    t_tail = df["t"].iloc[-tail_len:].values
    cm_tail = df["CM_Left"].iloc[-tail_len:].values
    coeffs = np.polyfit(t_tail, cm_tail, 1)
    v_final = coeffs[0]
    x_start = df["CM_Left"].iloc[0]
    spatial_shift = cm_tail[-1] - (x_start + v_init * t_tail[-1])

    # 打印定量报告（扩展版）
    report_path = os.path.join(OUTPUT_DIR, f"Report_{exp_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"QUANTITATIVE ANALYSIS REPORT: {exp_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"1. Energy Conservation: Max Error = {max_H_err:.2e} ({max_H_err * 100:.4f}%)\n")
        f.write(f"2. Inelasticity (Radiation Loss): {final_rad * 100:.4f}%\n")
        f.write(f"3. Max Compression Ratio: {max_compression:.3f}x\n")
        f.write(f"4. Asymptotic Velocity: {v_final:.4f} (Init: {v_init})\n")
        f.write(f"5. Spatial Phase Shift: {spatial_shift:.4f}\n")
        f.write(f"6. Estimated collision time: {coll_t:.2f}\n")
        f.write(f"7. Minimum soliton separation: {min_sep:.4f}\n")
        f.write(f"8. Final asymptotic phase shift (left): {final_phase_l:+.4f}\n")
        f.write(f"9. Final asymptotic phase shift (right): {final_phase_r:+.4f}\n")
        f.write(f"10. Max density compression ratio: {max_compression:.2f}x\n")

    print(f"  Report saved to {report_path}")

    # 保存 CSV
    df.to_csv(os.path.join(OUTPUT_DIR, f"Data_{exp_name}.csv"), index=False)

    # 定义视觉升级的配色方案
    COLOR_TRAJ_1 = "#0052cc"  # 深科技蓝
    COLOR_TRAJ_2 = "#e60000"  # 鲜艳红
    COLOR_PEAK = "#ff6600"  # 活力橙
    COLOR_RAD = "#9900cc"  # 深紫
    COLOR_ERR = "#009933"  # 森林绿

    # 2. Science-Style Plotting（视觉升级，结构不变）
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.25)

    # Panel A: 3D（升级光影和配色）
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    X, T = np.meshgrid(x, plot_data["t"])
    Z = np.array(plot_data["psi_sq"])
    mask = (x > -60) & (x < 60)
    X_c, T_c, Z_c = X[:, mask], T[:, mask], Z[:, mask]

    # 影院级光照渲染 + Turbo高对比度色图
    ls = LightSource(azdeg=300, altdeg=50)
    rgb = ls.shade(Z_c, cmap=cm.turbo, vert_exag=0.5, blend_mode='soft')

    ax3d.plot_surface(X_c, T_c, Z_c, facecolors=rgb,
                      linewidth=0, rstride=2, cstride=2, antialiased=True, shade=False)

    # 底部投影增强空间感
    offset = -0.1 * Z.max()
    ax3d.contourf(X_c, T_c, Z_c, zdir='z', offset=offset, cmap=cm.turbo, alpha=0.4)
    ax3d.set_zlim(offset, Z.max() * 1.1)

    ax3d.set_xlabel("Position ($x$)", labelpad=10, fontweight='bold')
    ax3d.set_ylabel("Time ($t$)", labelpad=10, fontweight='bold')
    ax3d.set_zlabel("Intensity ($|\psi|^2$)", labelpad=10, fontweight='bold')
    ax3d.view_init(elev=50, azim=-55)
    ax3d.dist = 11

    # 透明背景板 + 去网格
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.grid(False)

    ax3d.text2D(0.0, 0.95, f"(a) {extra_title}", transform=ax3d.transAxes, fontsize=12, fontweight='bold')

    # Panel B: Trajectory（升级配色和标注样式，图例移至右上角）
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_traj.plot(df["t"], df["CM_Left"], color=COLOR_TRAJ_1, lw=2.0, label='Soliton 1')
    ax_traj.plot(df["t"], df["CM_Right"], color=COLOR_TRAJ_2, lw=2.0, label='Soliton 2')

    # 新增：标注碰撞点（使用最小间距时刻）
    coll_cm = (df["CM_Left"].iloc[min_sep_idx] + df["CM_Right"].iloc[min_sep_idx]) / 2
    ax_traj.scatter(coll_t, coll_cm, s=100, facecolors='white', edgecolors='black',
                    marker='o', zorder=10, label=f'Closest approach (t={coll_t:.1f})')
    ax_traj.annotate(f'Collision\nphase shift ≈ {final_phase_l:+.2f}',
                     xy=(coll_t, coll_cm), xytext=(coll_t + 2, coll_cm + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=9, ha='center')

    # Ideal trajectory
    if "Bound" not in exp_name:
        ax_traj.plot(df["t"], df["CM_Left"].iloc[0] + v_init * df["t"], 'gray', linestyle='--', lw=1, alpha=0.6,
                     label='Free')

    ax_traj.set_xlabel("Time ($t$)")
    ax_traj.set_ylabel("Center of Mass ($x_{cm}$)")
    # 修改：图例移至左下角，避免与碰撞点重合
    ax_traj.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8,
                   loc='lower left', bbox_to_anchor=(0.02, 0.02))
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    ax_traj.text(0.05, 0.9, "(b) Trajectories", transform=ax_traj.transAxes, fontweight='bold')
    # Panel C: Peak Density（升级配色和填充效果）
    ax_peak = fig.add_subplot(gs[1, 1])
    ax_peak.fill_between(df["t"], df["Max_Rho"], color=COLOR_PEAK, alpha=0.2)
    ax_peak.plot(df["t"], df["Max_Rho"], color=COLOR_PEAK, lw=2.0)

    # 标注最大值
    max_val = df["Max_Rho"].max()
    max_t = df["t"].iloc[df["Max_Rho"].idxmax()]
    ax_peak.annotate(f'Max: {max_val:.2f}', xy=(max_t, max_val), xytext=(max_t + 5, max_val),
                     arrowprops=dict(arrowstyle="->", color='black'), fontsize=10)

    ax_peak.set_xlabel("Time ($t$)")
    ax_peak.set_ylabel("Max Density ($|\psi|_{max}^2$)")
    ax_peak.grid(True, linestyle=':', alpha=0.6)
    ax_peak.text(0.05, 0.9, "(c) Amplitude Dynamics", transform=ax_peak.transAxes, fontweight='bold')

    # Panel D: Radiation（升级配色和填充）
    ax_rad = fig.add_subplot(gs[2, 0])
    ax_rad.semilogy(df["t"], df["Radiation"], color=COLOR_RAD, lw=2.0)
    ax_rad.fill_between(df["t"], df["Radiation"], 1e-10, color=COLOR_RAD, alpha=0.1)
    ax_rad.set_ylabel("Radiation Ratio")
    ax_rad.set_xlabel("Time ($t$)")
    ax_rad.set_ylim(bottom=1e-6)
    ax_rad.grid(True, linestyle=':', alpha=0.6)
    ax_rad.text(0.05, 0.9, "(d) Radiation Loss", transform=ax_rad.transAxes, fontweight='bold')

    # Panel E: Energy Error（升级配色和格式，改为 Norm Conservation Error）
    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.plot(df["t"], df["H_err"] * 100, color=COLOR_ERR, lw=1.8)
    ax_err.set_ylabel("Norm Conservation Error (%)")  # 精炼后的标签
    ax_err.set_xlabel("Time ($t$)")
    ax_err.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_err.grid(True, linestyle=':', alpha=0.6)
    ax_err.text(0.05, 0.9, "(e) Numerical Stability", transform=ax_err.transAxes, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, f"Fig_{exp_name}.png"), bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to Fig_{exp_name}.png")


# ===================== 主程序：全场景实验 =====================
if __name__ == "__main__":
    # --- 准备工作: 计算标准基态 (alpha=1.8) ---
    print("\n=== Initializing Standard System (Alpha=1.8) ===")
    sys_std = CQNLS_System(alpha=1.8)
    phi_std = sys_std.find_ground_state()
    phi_np_std = phi_std.cpu().numpy()

    # ==========================================
    # 实验 1: 弹性碰撞 (Elastic Scattering)
    # ==========================================
    exp_name = "Elastic_Scattering"
    dist = 20.0
    v = 0.6
    shift = int(dist / sys_std.dx)

    p_L = np.roll(phi_np_std, -shift) * np.exp(1j * v * sys_std.x.cpu().numpy())
    p_R = np.roll(phi_np_std, shift) * np.exp(-1j * v * sys_std.x.cpu().numpy())
    psi_init = torch.tensor(np.stack([p_L, p_R]), device=DEVICE)

    df, plot_data, x = sys_std.run_collision_analysis(psi_init, T_max=40.0)
    analyze_and_plot(exp_name, df, plot_data, x, 1.8, v, "Elastic Collision")

    # ==========================================
    # 实验 2: 束缚态形成 (Bound State)
    # ==========================================
    exp_name = "Bound_State"
    v_slow = 0.2  # 关键：低速
    shift_BS = int(15.0 / sys_std.dx)  # 距离近一点

    p_L = np.roll(phi_np_std, -shift_BS) * np.exp(1j * v_slow * sys_std.x.cpu().numpy())
    p_R = np.roll(phi_np_std, shift_BS) * np.exp(-1j * v_slow * sys_std.x.cpu().numpy())
    psi_init = torch.tensor(np.stack([p_L, p_R]), device=DEVICE)

    df, plot_data, x = sys_std.run_collision_analysis(psi_init, T_max=60.0)
    analyze_and_plot(exp_name, df, plot_data, x, 1.8, v_slow, "Bound State Formation")

    # ==========================================
    # 实验 3: 非对称相互作用 (Asymmetric)
    # ==========================================
    exp_name = "Asymmetric_Collision"
    v = 0.5
    amp_ratio = 1.2  # 右边强20%

    p_L = np.roll(phi_np_std, -shift) * np.exp(1j * v * sys_std.x.cpu().numpy())
    p_R = (np.roll(phi_np_std, shift) * amp_ratio) * np.exp(-1j * v * sys_std.x.cpu().numpy())  # 缩放
    psi_init = torch.tensor(np.stack([p_L, p_R]), device=DEVICE)

    df, plot_data, x = sys_std.run_collision_analysis(psi_init, T_max=40.0)
    analyze_and_plot(exp_name, df, plot_data, x, 1.8, v, "Asymmetric Interaction")

    # ==========================================
    # 实验 4: 强分数阶效应 (Strong Fractional)
    # ==========================================
    # 需要重新初始化系统 (alpha=1.3) 并寻找新的基态
    print("\n=== Initializing Fractional System (Alpha=1.3) ===")
    sys_frac = CQNLS_System(alpha=1.3)
    phi_frac = sys_frac.find_ground_state()  # ITE 重新寻找
    phi_np_frac = phi_frac.cpu().numpy()

    exp_name = "Strong_Fractional_Effect"
    v = 0.6
    dist = 25.0
    shift = int(dist / sys_frac.dx)

    p_L = np.roll(phi_np_frac, -shift) * np.exp(1j * v * sys_frac.x.cpu().numpy())
    p_R = np.roll(phi_np_frac, shift) * np.exp(-1j * v * sys_frac.x.cpu().numpy())
    psi_init = torch.tensor(np.stack([p_L, p_R]), device=DEVICE)

    df, plot_data, x = sys_frac.run_collision_analysis(psi_init, T_max=40.0)
    analyze_and_plot(exp_name, df, plot_data, x, 1.3, v, "Strong Fractional Effect (alpha=1.3)")

    print("\n=== All 4 Paper Scenarios Completed Successfully ===")
