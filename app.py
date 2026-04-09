"""
================================================
AI RISK DASHBOARD — MULTI-SECTOR AUDIT TOOL
================================================
Author  : [Your Name]
Project : Astra Fellowship Portfolio — AI Safety
Sectors : Healthcare · Credit & Lending · Criminal Justice
================================================
Run: streamlit run app.py
================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io, textwrap, datetime, os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Risk Dashboard",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #7C3AED 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.3rem;
}
.hero-sub { color: #64748B; font-size: 1.05rem; margin-bottom: 1.5rem; }

.risk-card-critical {
    background: linear-gradient(135deg, #FEF2F2, #FFF); 
    border-left: 5px solid #DC2626; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin: 0.6rem 0;
    box-shadow: 0 2px 8px rgba(220,38,38,0.12);
}
.risk-card-high {
    background: linear-gradient(135deg, #FFFBEB, #FFF);
    border-left: 5px solid #D97706; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin: 0.6rem 0;
    box-shadow: 0 2px 8px rgba(217,119,6,0.12);
}
.risk-card-medium {
    background: linear-gradient(135deg, #EFF6FF, #FFF);
    border-left: 5px solid #2563EB; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin: 0.6rem 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.10);
}
.risk-card-low {
    background: linear-gradient(135deg, #F0FDF4, #FFF);
    border-left: 5px solid #16A34A; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin: 0.6rem 0;
    box-shadow: 0 2px 8px rgba(22,163,74,0.10);
}
.section-title {
    font-size: 1.25rem; font-weight: 700; color: #1E3A5F;
    border-bottom: 2px solid #2563EB; padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem 0;
}
.plain-english {
    background: #F8FAFF; border: 1px solid #DBEAFE;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin: 0.8rem 0;
    font-size: 0.97rem; line-height: 1.7;
}
.score-badge-critical { color: #DC2626; font-size: 2rem; font-weight: 800; }
.score-badge-high     { color: #D97706; font-size: 2rem; font-weight: 800; }
.score-badge-medium   { color: #2563EB; font-size: 2rem; font-weight: 800; }
.score-badge-low      { color: #16A34A; font-size: 2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
PALETTE = {
    'critical': '#DC2626', 'high': '#D97706',
    'medium':   '#2563EB', 'low':  '#16A34A',
    'neutral':  '#6B7280', 'purple': '#7C3AED',
    'bg':       '#F8FAFC',
}
sns.set_theme(style='whitegrid', font_scale=1.05)

SECTOR_META = {
    "🏥 Healthcare": {
        "file": "data/healthcare_dataset.csv",
        "outcome_col": "ai_referred_specialist",
        "score_col":   "ai_risk_score",
        "outcome_label": "Specialist Referral",
        "protected":   ["gender", "race", "income_level", "insurance_status"],
        "legitimate":  ["age", "bmi", "systolic_bp", "glucose", "smoking", "family_history"],
        "privileged":  {"race": "White", "gender": "Male",
                        "income_level": "High", "insurance_status": "Insured"},
        "description": "An AI triage system that decides which patients get referred to a specialist.",
        "stakes": "🚨 HIGH STAKES — Missed referrals can lead to delayed diagnosis, disease progression, or death.",
        "real_world": "Systems like this are used in hospital networks across the US and are being piloted in several African countries.",
    },
    "💳 Credit & Lending": {
        "file": "data/credit_dataset.csv",
        "outcome_col": "loan_approved",
        "score_col":   "ai_risk_score",
        "outcome_label": "Loan Approval",
        "protected":   ["gender", "race", "zip_code_risk"],
        "legitimate":  ["credit_score", "debt_to_income", "employment_years",
                        "loan_amount_requested", "previous_defaults"],
        "privileged":  {"race": "White", "gender": "Male", "zip_code_risk": "Low"},
        "description": "An AI credit scoring model that approves or rejects loan applications.",
        "stakes": "🚨 HIGH STAKES — Unfair rejections lock people out of homeownership, education, and business.",
        "real_world": "AI credit models now process over 80% of consumer loan decisions globally.",
    },
    "⚖️ Criminal Justice": {
        "file": "data/criminal_justice_dataset.csv",
        "outcome_col": "ai_recommends_detention",
        "score_col":   "ai_recidivism_score",
        "outcome_label": "Detention Recommended",
        "protected":   ["gender", "race"],
        "legitimate":  ["prior_offenses", "charge_severity", "employed",
                        "education_level", "community_ties"],
        "privileged":  {"race": "White", "gender": "Female"},
        "description": "An AI recidivism risk model used to recommend pre-trial detention.",
        "stakes": "🔴 CRITICAL STAKES — Wrongful detention destroys lives, livelihoods, and families.",
        "real_world": "COMPAS and similar tools are used in courts across the US; similar tools are being considered in Africa.",
    },
}

# ── Helpers ───────────────────────────────────────────────────
@st.cache_data
def load_sector(path):
    return pd.read_csv(path)

def disparate_impact(df, group_col, outcome_col, privileged):
    rates = df.groupby(group_col)[outcome_col].mean()
    ref   = rates.get(privileged, rates.mean())
    out   = {}
    for g, r in rates.items():
        if outcome_col == 'loan_approved':
            di = r / ref if ref > 0 else 1.0
        else:
            di = r / ref if ref > 0 else 1.0
        out[g] = {'rate': round(r, 4), 'di': round(di, 4)}
    return out

def risk_level(di_val, is_ref=False):
    if is_ref: return 'medium', '⚡ Reference Group'
    if di_val >= 0.90: return 'low',      '✅ FAIR'
    if di_val >= 0.80: return 'medium',   '⚠️ BORDERLINE'
    if di_val >= 0.60: return 'high',     '🚨 BIASED'
    return 'critical', '🔴 SEVERELY BIASED'

def score_color(score):
    if score >= 75: return 'critical'
    if score >= 55: return 'high'
    if score >= 35: return 'medium'
    return 'low'

def plain_english_bias(group_col, group, di, rate, outcome_label, privileged, outcome_col):
    pct = rate * 100
    di_pct = di * 100
    if outcome_col == 'loan_approved':
        direction = "approved" if di >= 1 else "approved"
        gap = abs(1 - di) * 100
        if di < 0.80:
            return (f"The AI approves loans for <b>{group}</b> applicants at only "
                    f"<b>{pct:.1f}%</b> of the rate it approves {privileged} applicants — "
                    f"a <b>{gap:.0f}% gap</b>. This means equally qualified {group} borrowers "
                    f"face systematic rejection. Under US law, this gap alone can constitute illegal discrimination.")
        else:
            return (f"The AI approves <b>{group}</b> applicants at <b>{pct:.1f}%</b> — "
                    f"close to the reference group rate. No significant bias detected for this group.")
    elif outcome_col == 'ai_referred_specialist':
        gap = abs(1 - di) * 100
        if di < 0.80:
            return (f"The AI refers <b>{group}</b> patients to specialists <b>{gap:.0f}% less often</b> "
                    f"than {privileged} patients with equivalent symptoms. "
                    f"This means sicker {group} patients may be sent home without proper care — "
                    f"a direct patient safety risk.")
        else:
            return (f"The AI refers <b>{group}</b> patients at a <b>similar rate</b> to {privileged} patients ({pct:.1f}%). "
                    f"No significant referral bias detected.")
    else:
        gap = abs(1 - di) * 100
        if di > 1.20:
            return (f"The AI recommends detaining <b>{group}</b> individuals at "
                    f"<b>{gap:.0f}% higher rates</b> than {privileged} individuals with identical charges. "
                    f"This is the AI equivalent of systemic racial profiling — people lose their freedom "
                    f"not because of what they did, but because of who they are.")
        elif di < 0.80:
            return (f"The AI recommends detaining <b>{group}</b> individuals at "
                    f"<b>{gap:.0f}% lower rates</b> than average.")
        else:
            return (f"The AI treats <b>{group}</b> individuals at a <b>similar rate</b> to {privileged} ({pct:.1f}%). "
                    f"No significant bias detected.")

def compute_overall_risk_score(di_results_all):
    """Compute an overall bias risk score 0–100."""
    all_di = []
    for group_col, results in di_results_all.items():
        for g, vals in results.items():
            all_di.append(vals['di'])
    if not all_di:
        return 50
    worst_di = min(all_di)
    avg_di   = np.mean(all_di)
    # Score: 0 = perfect fair, 100 = maximally biased
    score = max(0, min(100, (1 - worst_di) * 80 + (1 - avg_di) * 20)) * 100
    return round(min(score, 100), 1)

def make_gauge_fig(score):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor('#F8FAFC')
    # Background arc
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#E2E8F0', linewidth=18, solid_capstyle='round')
    # Colored arc
    frac  = score / 100
    theta2= np.linspace(np.pi, np.pi - frac * np.pi, 300)
    color = PALETTE['critical'] if score >= 75 else PALETTE['high'] if score >= 55 \
            else PALETTE['medium'] if score >= 35 else PALETTE['low']
    ax.plot(np.cos(theta2), np.sin(theta2), color=color, linewidth=18, solid_capstyle='round')
    ax.text(0, -0.15, f"{score:.0f}", ha='center', va='center',
            fontsize=32, fontweight='800', color=color)
    ax.text(0, -0.52, "BIAS RISK SCORE", ha='center', va='center',
            fontsize=8, fontweight='600', color='#64748B')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.7, 1.15)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def make_di_chart(di_results, group_col, outcome_col, privileged):
    groups = list(di_results.keys())
    di_vals= [di_results[g]['di'] for g in groups]
    rates  = [di_results[g]['rate'] * 100 for g in groups]
    colors = []
    for g, dv in zip(groups, di_vals):
        if g == privileged:  colors.append(PALETTE['medium'])
        elif dv >= 0.80:     colors.append(PALETTE['low'])
        elif dv >= 0.60:     colors.append(PALETTE['high'])
        else:                colors.append(PALETTE['critical'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#F8FAFC')

    # Left: Outcome rates
    bars = axes[0].bar(groups, rates, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    axes[0].set_title(f'Outcome Rate by {group_col.replace("_"," ").title()}', fontweight='bold')
    axes[0].set_ylabel('Rate (%)')
    mean_rate = np.mean(rates)
    axes[0].axhline(mean_rate, color=PALETTE['neutral'], linestyle='--',
                    linewidth=1.5, label=f'Average ({mean_rate:.1f}%)')
    axes[0].legend(fontsize=9)
    for b, v in zip(bars, rates):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[0].set_xticklabels(groups, rotation=20, ha='right')

    # Right: Disparate Impact
    bars2 = axes[1].bar(groups, di_vals, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    axes[1].axhline(0.80, color=PALETTE['critical'], linestyle='--',
                    linewidth=2, label='Fairness Threshold (0.80)')
    axes[1].axhline(1.00, color=PALETTE['neutral'], linestyle=':', linewidth=1.2, alpha=0.6)
    axes[1].set_title(f'Disparate Impact Ratio (vs {privileged})', fontweight='bold')
    axes[1].set_ylabel('Disparate Impact Ratio')
    axes[1].set_ylim(0, max(1.5, max(di_vals) + 0.2))
    axes[1].legend(fontsize=9)
    for b, v in zip(bars2, di_vals):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                     f'{v:.2f}', ha='center', fontsize=9, fontweight='bold',
                     color=PALETTE['critical'] if v < 0.80 else PALETTE['low'])
    axes[1].set_xticklabels(groups, rotation=20, ha='right')

    plt.tight_layout()
    return fig

def make_distribution_fig(df, score_col, outcome_col, group_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#F8FAFC')
    groups = sorted(df[group_col].unique())
    palette = sns.color_palette("husl", len(groups))

    for i, (grp, col) in enumerate(zip(groups, palette)):
        sub = df[df[group_col] == grp][score_col]
        axes[0].hist(sub, bins=30, alpha=0.5, color=col, label=grp, density=True)
    axes[0].set_title(f'AI Score Distribution by {group_col.replace("_"," ").title()}',
                      fontweight='bold')
    axes[0].set_xlabel('AI Risk Score'); axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8)

    mean_scores = df.groupby(group_col)[score_col].mean().sort_values()
    colors_m = [PALETTE['critical'] if v > 0.6 else PALETTE['high'] if v > 0.5
                else PALETTE['low'] for v in mean_scores.values]
    bars = axes[1].barh(mean_scores.index, mean_scores.values,
                        color=colors_m, edgecolor='white', height=0.55)
    axes[1].set_title(f'Mean AI Score by {group_col.replace("_"," ").title()}',
                      fontweight='bold')
    axes[1].set_xlabel('Mean AI Risk Score')
    for b, v in zip(bars, mean_scores.values):
        axes[1].text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
                     f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    return fig

def make_data_quality_fig(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#F8FAFC')
    numeric = df.select_dtypes(include=[np.number])

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=True)
    if missing_pct.sum() == 0:
        axes[0].text(0.5, 0.5, '✅ No Missing Values\nData Quality: PASS',
                     ha='center', va='center', fontsize=13, fontweight='bold',
                     color=PALETTE['low'], transform=axes[0].transAxes)
    else:
        axes[0].barh(missing_pct.index, missing_pct.values, color=PALETTE['critical'])
    axes[0].set_title('Missing Values (%)', fontweight='bold')
    axes[0].set_xlabel('% Missing')

    # Outlier detection (IQR method)
    outlier_counts = {}
    for col in numeric.columns[:8]:
        Q1, Q3 = numeric[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_counts[col] = ((numeric[col] < Q1 - 1.5*IQR) |
                               (numeric[col] > Q3 + 1.5*IQR)).sum()
    oc = pd.Series(outlier_counts).sort_values(ascending=True)
    colors_o = [PALETTE['critical'] if v > 50 else PALETTE['high'] if v > 20
                else PALETTE['low'] for v in oc.values]
    axes[1].barh(oc.index, oc.values, color=colors_o, edgecolor='white', height=0.6)
    axes[1].set_title('Outliers per Feature (IQR Method)', fontweight='bold')
    axes[1].set_xlabel('Count of Outliers')

    # Class balance
    outcome_cols = [c for c in df.columns if c in
                    ['ai_referred_specialist','loan_approved','ai_recommends_detention','high_risk_flag']]
    if outcome_cols:
        oc_col = outcome_cols[0]
        counts = df[oc_col].value_counts()
        colors_b = [PALETTE['medium'], PALETTE['critical']] if counts[0] > counts.iloc[-1]*3 \
                   else [PALETTE['medium'], PALETTE['low']]
        bars = axes[2].bar(['Class 0\n(Negative)', 'Class 1\n(Positive)'],
                           counts.values, color=[PALETTE['low'], PALETTE['critical']],
                           edgecolor='white', width=0.5)
        ratio = counts.values[0] / max(counts.values[1], 1)
        axes[2].set_title(f'Class Balance\nImbalance Ratio: {ratio:.1f}:1', fontweight='bold')
        axes[2].set_ylabel('Count')
        for b, v in zip(bars, counts.values):
            axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+20,
                         f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_pdf_report(sector_name, meta, df, di_all, overall_score):
    """Generate a plain-text report (markdown format) for download."""
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 65)
    lines.append("   AI RISK AUDIT REPORT")
    lines.append(f"   Sector      : {sector_name}")
    lines.append(f"   Generated   : {now}")
    lines.append(f"   Records     : {len(df):,}")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Overall Bias Risk Score: {overall_score:.0f} / 100")
    level = ("CRITICAL" if overall_score >= 75 else "HIGH" if overall_score >= 55
             else "MEDIUM" if overall_score >= 35 else "LOW")
    lines.append(f"Risk Level: {level}")
    lines.append("")
    lines.append(meta['description'])
    lines.append("")
    lines.append(f"Stakes: {meta['stakes']}")
    lines.append(f"Real-World Context: {meta['real_world']}")
    lines.append("")

    lines.append("BIAS FINDINGS BY PROTECTED GROUP")
    lines.append("-" * 40)
    lines.append("(Threshold: Disparate Impact >= 0.80 = FAIR per EEOC 4/5ths Rule)")
    lines.append("")
    for group_col, results in di_all.items():
        lines.append(f"  Protected Attribute: {group_col.upper()}")
        privileged = meta['privileged'].get(group_col, '')
        for g, vals in sorted(results.items(), key=lambda x: x[1]['di']):
            di   = vals['di']
            rate = vals['rate'] * 100
            flag = ("FAIR" if di >= 0.80 else "BIASED" if di >= 0.60 else "SEVERELY BIASED")
            marker = "  " if g == privileged else ("OK  " if di >= 0.80 else "WARN")
            lines.append(f"    [{marker}] {g:20s}: Rate={rate:.1f}%  DI={di:.3f}  {flag}")
        lines.append("")

    lines.append("DATA QUALITY SUMMARY")
    lines.append("-" * 40)
    missing_total = df.isnull().sum().sum()
    lines.append(f"  Missing values     : {missing_total}")
    lines.append(f"  Total records      : {len(df):,}")
    outcome_col = meta['outcome_col']
    if outcome_col in df.columns:
        pos_rate = df[outcome_col].mean() * 100
        lines.append(f"  Positive outcome % : {pos_rate:.1f}%")
        ratio = (1 - df[outcome_col].mean()) / max(df[outcome_col].mean(), 0.001)
        lines.append(f"  Class imbalance    : {ratio:.1f}:1")
    lines.append("")

    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append("  1. IMMEDIATE: Commission a third-party algorithmic audit")
    lines.append("  2. Remove protected attributes from model input features")
    lines.append("  3. Apply reweighing or adversarial debiasing techniques")
    lines.append("  4. Implement continuous fairness monitoring (monthly DI checks)")
    lines.append("  5. Establish human-in-the-loop review for borderline cases")
    lines.append("  6. Document model decisions (model cards) for transparency")
    lines.append("  7. Engage affected communities in model design and review")
    lines.append("")

    lines.append("FAIRNESS METRICS GLOSSARY")
    lines.append("-" * 40)
    lines.append("  Disparate Impact (DI): Ratio of outcome rates between groups.")
    lines.append("    DI = P(outcome | unprivileged) / P(outcome | privileged)")
    lines.append("    EEOC 4/5ths Rule: DI < 0.80 is legally discriminatory.")
    lines.append("  Statistical Parity: Groups should have equal outcome rates.")
    lines.append("  Equal Opportunity: Groups should have equal true positive rates.")
    lines.append("")
    lines.append("=" * 65)
    lines.append("  Generated by AI Risk Dashboard | Astra Fellowship Portfolio")
    lines.append("  This report is for educational and auditing purposes.")
    lines.append("=" * 65)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚠️ AI Risk Dashboard")
    st.markdown("*Multi-Sector AI Bias Auditor*")
    st.markdown("---")
    sector = st.selectbox(
        "Select a Sector to Audit:",
        list(SECTOR_META.keys()),
        index=0
    )
    st.markdown("---")
    page = st.radio("Section:", [
        "🏠 Overview & Stakes",
        "📊 Bias Audit",
        "🔍 Data Quality",
        "📋 Risk Report",
    ])
    st.markdown("---")
    st.markdown("**Fairness Standard Used:**")
    st.markdown("EEOC 4/5ths Rule — Disparate Impact ≥ 0.80")
    st.markdown("---")
    st.caption("Astra Fellowship Portfolio | AI Safety Project")

# Load sector data
meta = SECTOR_META[sector]
df   = load_sector(meta['file'])
outcome_col = meta['outcome_col']
score_col   = meta['score_col']

# Compute all DI results
di_all = {}
for gc in meta['protected']:
    if gc in df.columns:
        priv = meta['privileged'].get(gc, df[gc].mode()[0])
        di_all[gc] = disparate_impact(df, gc, outcome_col, priv)

overall_score = compute_overall_risk_score(di_all)
sc = score_color(overall_score)

# ═══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & STAKES
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview & Stakes":
    st.markdown(f'<div class="hero-title">AI Risk Audit: {sector}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{meta["description"]}</div>', unsafe_allow_html=True)

    col_gauge, col_info = st.columns([1, 2])
    with col_gauge:
        gauge_fig = make_gauge_fig(overall_score)
        st.pyplot(gauge_fig, use_container_width=False)
        plt.close()
        label = ("CRITICAL RISK" if sc == 'critical' else "HIGH RISK" if sc == 'high'
                 else "MEDIUM RISK" if sc == 'medium' else "LOW RISK")
        st.markdown(f'<div style="text-align:center; font-weight:800; font-size:1.1rem; color:{PALETTE[sc]}">{label}</div>',
                    unsafe_allow_html=True)

    with col_info:
        st.markdown(f'<div class="risk-card-{sc}"><b>⚠️ What This System Does</b><br><br>{meta["description"]}<br><br><b>{meta["stakes"]}</b><br><br>📌 {meta["real_world"]}</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset at a Glance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Records", f"{len(df):,}")
    with c2: st.metric("Protected Attributes", str(len(meta['protected'])))
    with c3:
        pos_rate = df[outcome_col].mean() * 100
        st.metric(meta['outcome_label'] + " Rate", f"{pos_rate:.1f}%")
    with c4: st.metric("Bias Risk Score", f"{overall_score:.0f} / 100")

    st.markdown('<div class="section-title">Why This Matters — Plain English</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="plain-english">
    <b>Imagine this:</b> An AI system is making decisions that affect people's health, money, or freedom.
    It was trained on historical data — data that reflects decades of human bias and inequality.<br><br>
    The system doesn't know it's being unfair. It's just doing math. But the result is the same:
    <b>some groups of people are systematically treated worse than others</b> — not because of anything
    they did, but because of who they are.<br><br>
    This dashboard audits the AI system in the <b>{sector}</b> sector and answers three questions:
    <ol>
    <li>🔍 <b>Is the AI treating different groups differently?</b></li>
    <li>📏 <b>How bad is the disparity — and is it legally discriminatory?</b></li>
    <li>🛡️ <b>What should be done about it?</b></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Bias Snapshot</div>', unsafe_allow_html=True)
    for gc in meta['protected']:
        if gc not in di_all:
            continue
        priv = meta['privileged'].get(gc, '')
        worst_group = min([(g, v['di']) for g, v in di_all[gc].items() if g != priv],
                          key=lambda x: x[1], default=(None, 1.0))
        if worst_group[0]:
            wg, wdi = worst_group
            lv, flag = risk_level(wdi)
            worst_rate = di_all[gc][wg]['rate'] * 100
            priv_rate  = di_all[gc].get(priv, {}).get('rate', 0) * 100
            st.markdown(f'<div class="risk-card-{lv}"><b>{flag} &nbsp; {gc.replace("_"," ").title()}</b><br>'
                        f'Most affected group: <b>{wg}</b> — outcome rate <b>{worst_rate:.1f}%</b> '
                        f'vs <b>{priv_rate:.1f}%</b> for {priv} (Disparate Impact: {wdi:.2f})</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: BIAS AUDIT
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Bias Audit":
    st.markdown(f'<div class="hero-title">Bias Audit — {sector}</div>', unsafe_allow_html=True)
    st.markdown("Measuring how equally the AI treats different groups of people.")

    tabs = st.tabs([f"📌 {gc.replace('_',' ').title()}" for gc in meta['protected'] if gc in df.columns])

    for tab, gc in zip(tabs, [g for g in meta['protected'] if g in df.columns]):
        with tab:
            priv    = meta['privileged'].get(gc, df[gc].mode()[0])
            results = di_all[gc]

            st.markdown(f"**Reference group (most advantaged): {priv}**")
            st.markdown("---")

            # Cards for each group
            for g in sorted(results.keys(), key=lambda x: results[x]['di']):
                vals = results[g]
                di   = vals['di']
                rate = vals['rate']
                is_ref = (g == priv)
                lv, flag = risk_level(di, is_ref)
                explanation = plain_english_bias(gc, g, di, rate, meta['outcome_label'],
                                                  priv, outcome_col)
                st.markdown(
                    f'<div class="risk-card-{lv}">'
                    f'<b>{flag} &nbsp; {g}</b>'
                    f'<span style="float:right; font-size:1.4rem; font-weight:800; color:{PALETTE[lv]}">'
                    f'DI = {di:.3f}</span><br><br>'
                    f'<span style="color:#374151">{explanation}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)
            fig = make_di_chart(results, gc, outcome_col, priv)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
            fig2 = make_distribution_fig(df, score_col, outcome_col, gc)
            st.pyplot(fig2, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE 3: DATA QUALITY
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality":
    st.markdown(f'<div class="hero-title">Data Quality Audit — {sector}</div>', unsafe_allow_html=True)
    st.markdown("Bad data produces bad AI. This section checks whether the underlying data is trustworthy.")

    st.markdown('<div class="section-title">Quality Checks</div>', unsafe_allow_html=True)
    fig = make_data_quality_fig(df)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Quality cards
    missing = df.isnull().sum().sum()
    lv_miss = 'low' if missing == 0 else 'high'
    st.markdown(f'<div class="risk-card-{lv_miss}"><b>Missing Data:</b> '
                f'{"No missing values found — data completeness is PASS ✅" if missing == 0 else f"{missing} missing values detected 🚨 — these may introduce bias if not handled carefully."}'
                f'</div>', unsafe_allow_html=True)

    pos_rate = df[outcome_col].mean()
    imbalance_ratio = (1 - pos_rate) / max(pos_rate, 0.001)
    lv_bal = 'low' if imbalance_ratio < 5 else 'high' if imbalance_ratio > 10 else 'medium'
    st.markdown(f'<div class="risk-card-{lv_bal}"><b>Class Balance:</b> '
                f'The dataset has a {imbalance_ratio:.1f}:1 imbalance. '
                f'{"This is manageable — class_weight=balanced handles this." if lv_bal == "low" else "High imbalance — the model may struggle to learn minority class patterns and produce biased predictions for underrepresented groups."}'
                f'</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sample Data</div>', unsafe_allow_html=True)
    st.markdown("*First 50 records:*")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: RISK REPORT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Risk Report":
    st.markdown(f'<div class="hero-title">AI Risk Audit Report — {sector}</div>', unsafe_allow_html=True)
    st.markdown("A complete, plain-English summary of all findings. Download as a text report.")

    # ── Summary panel ──────────────────────────────────────────
    lv = score_color(overall_score)
    label = ("CRITICAL RISK" if lv == 'critical' else "HIGH RISK" if lv == 'high'
             else "MEDIUM RISK" if lv == 'medium' else "LOW RISK")

    col_s, col_g = st.columns([2, 1])
    with col_s:
        st.markdown(f"""
        <div class="risk-card-{lv}">
        <b style="font-size:1.1rem">Overall Bias Risk Score: {overall_score:.0f} / 100 — {label}</b><br><br>
        This AI system in the <b>{sector}</b> sector shows <b>{"significant" if overall_score >= 55 else "moderate" if overall_score >= 35 else "low"}</b>
        levels of algorithmic bias across protected demographic groups.
        {"Immediate action is required before this system can be considered safe for deployment." if overall_score >= 75
        else "Remediation steps should be taken before wider deployment." if overall_score >= 55
        else "The system shows some bias that should be addressed through standard fairness techniques." if overall_score >= 35
        else "The system appears relatively fair, but continuous monitoring is still recommended."}
        </div>
        """, unsafe_allow_html=True)
    with col_g:
        gf = make_gauge_fig(overall_score)
        st.pyplot(gf, use_container_width=False)
        plt.close()

    # ── Per-group findings ─────────────────────────────────────
    st.markdown('<div class="section-title">Findings by Protected Group</div>', unsafe_allow_html=True)
    for gc in meta['protected']:
        if gc not in di_all:
            continue
        priv    = meta['privileged'].get(gc, '')
        results = di_all[gc]
        st.markdown(f"#### {gc.replace('_',' ').title()}")
        for g in sorted(results.keys(), key=lambda x: results[x]['di']):
            vals = results[g]
            di   = vals['di']
            rate = vals['rate']
            is_ref = (g == priv)
            lv2, flag = risk_level(di, is_ref)
            explanation = plain_english_bias(gc, g, di, rate, meta['outcome_label'],
                                              priv, outcome_col)
            st.markdown(
                f'<div class="risk-card-{lv2}" style="margin-bottom:0.4rem">'
                f'<b>{flag} {g}</b> &nbsp;|&nbsp; DI = {di:.3f} &nbsp;|&nbsp; '
                f'Outcome rate: {rate*100:.1f}%<br>'
                f'<span style="font-size:0.93rem;color:#374151">{explanation}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("🔴 Immediate", "Commission a third-party algorithmic audit before this system processes any more decisions."),
        ("🔴 Immediate", "Notify relevant regulators and internal compliance teams of these findings."),
        ("🟠 Short-term", "Remove protected attributes (race, gender, etc.) from the model's input features."),
        ("🟠 Short-term", "Apply sample reweighing or adversarial debiasing to retrain the model."),
        ("🟡 Medium-term", "Implement monthly Disparate Impact monitoring with automatic alerts."),
        ("🟡 Medium-term", "Build a human-in-the-loop review process for borderline decisions."),
        ("🟢 Long-term", "Publish a model card — a transparency document that explains what the model does and its known limitations."),
        ("🟢 Long-term", "Engage affected communities in the model design and audit process."),
    ]
    for priority, rec in recs:
        lv3 = ('critical' if '🔴' in priority else 'high' if '🟠' in priority
               else 'medium' if '🟡' in priority else 'low')
        st.markdown(f'<div class="risk-card-{lv3}"><b>{priority}:</b> {rec}</div>',
                    unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
    report_text = generate_pdf_report(sector, meta, df, di_all, overall_score)
    st.download_button(
        label="📥 Download Full Audit Report (.txt)",
        data=report_text.encode('utf-8'),
        file_name=f"ai_risk_audit_{sector.replace(' ','_').replace('🏥','health').replace('💳','credit').replace('⚖️','justice')}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True
    )
    st.caption("The report is a plain-text audit document. Share it with your compliance team, regulator, or leadership.")
