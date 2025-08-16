import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---- 放最顶：页面配置 ----
st.set_page_config(page_title="心脏病预测 · Streamlit", page_icon="❤️", layout="centered")
st.title("❤️ 心脏病预测（Streamlit 版）")
st.caption("建议：统一使用列名 **thalch**（最大心率）。本应用优先加载仓库根目录下的模型文件。")

# ---- ① 加载模型（只保留这一套逻辑：候选文件 -> 上传兜底） ----
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [BASE_DIR / "best_model_pipe.pkl", BASE_DIR / "best_model.pkl"]

st.subheader("① 加载模型")

def try_load(path: Path):
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, e  # 不崩溃，交给 UI 展示

loaded, err, src = None, None, None
for p in CANDIDATES:
    if p.exists():
        loaded, err = try_load(p)
        src = p
        break

if loaded is None:
    st.warning("未在仓库根目录找到模型文件，或加载失败。你也可以上传一个：")
    up = st.file_uploader("上传模型（.pkl）", type=["pkl"])
    if up:
        try:
            loaded = joblib.load(up)
            src = up.name
        except Exception as e:
            err = e

st.write("📄 模型来源：", str(src) if src else "（未找到）")
st.write("🧠 对象类型：", type(loaded).__name__ if loaded is not None else "None")

if loaded is None:
    st.error(f"模型加载失败：{err}")
    st.stop()

if not hasattr(loaded, "predict"):
    st.error(f"当前加载对象不是可预测的模型/Pipeline。类型：{type(loaded)}")
    st.stop()

model = loaded

# ---- ② 定义输入特征 ----
st.subheader("② 定义输入特征")
feature_names = list(getattr(model, "feature_names_in_", []))
csv_df = None
if not feature_names:
    st.info("模型未提供 feature_names_in_。请上传一个示例 CSV 用于构建表单。")
    csv_up = st.file_uploader("上传示例 CSV（列名将用于生成输入表单）", type=["csv"], key="csv_schema")
    if csv_up:
        try:
            csv_df = pd.read_csv(csv_up)
            feature_names = list(csv_df.columns)
        except Exception as e:
            st.error(f"CSV 读取失败：{e}")

if not feature_names:
    st.error("无法确定特征名。请：1）使用带 feature_names_in_ 的模型导出；或 2）提供包含完整列名的 CSV。")
    st.stop()

# ---- ③ 构建输入表单 ----
st.subheader("③ 填写特征值")
st.sidebar.header("填写特征")

def build_default(s: pd.Series):
    if s is None or s.empty:
        return 0
    if pd.api.types.is_numeric_dtype(s):
        return float(s.median())
    m = s.mode()
    return (m.iloc[0] if not m.empty else "")

schema_df = csv_df
def widget_for(c):
    s = schema_df[c] if (schema_df is not None and c in schema_df.columns) else None
    is_num = pd.api.types.is_numeric_dtype(s) if s is not None else True
    if is_num:
        if s is not None and not s.empty:
            q01, q99 = float(s.quantile(0.01)), float(s.quantile(0.99))
            default = build_default(s)
            step = 1.0 if s.dropna().apply(lambda x: float(x).is_integer()).all() else 0.1
        else:
            q01, q99, default, step = 0.0, 100.0, 0.0, 1.0
        return st.sidebar.number_input(c, min_value=q01, max_value=q99, value=float(default), step=step, key=f"n_{c}")
    else:
        opts = s.dropna().unique().tolist() if (s is not None and not s.empty) else []
        if 0 < len(opts) <= 15:
            return st.sidebar.selectbox(c, options=opts or [""], key=f"s_{c}")
        return st.sidebar.text_input(c, value=str(build_default(s)), key=f"t_{c}")

inputs = {c: widget_for(c) for c in feature_names}
X = pd.DataFrame([inputs], columns=feature_names)
st.markdown("**你的输入**")
st.dataframe(X, use_container_width=True)

# ---- ④ 预测 ----
st.subheader("④ 预测")
if st.sidebar.button("开始预测", type="primary", use_container_width=True):
    try:
        y = model.predict(X)
        st.success(f"预测类别：**{y[0]}**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                st.info(f"阳性概率：**{proba[0,1]:.3f}**；阴性概率：{proba[0,0]:.3f}")
            else:
                st.write("各类别概率：")
                st.dataframe(pd.Series(proba[0]).to_frame("probability"))
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.caption("通常是列名或类型与训练不一致；请统一使用 thalch 作为最大心率列名。")
