# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
from io import BytesIO
# =======================
# إعداد الصفحة (Page Config)
# =======================
st.set_page_config(
    page_title="Student Scores ML Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =======================
# وظائف مساعدة
# =======================
TARGETS = ["MathScore", "ReadingScore", "WritingScore"]
@st.cache_data(show_spinner=False)
def load_data(default_path: str = "/mnt/data/Expanded_data_with_more_features.csv"):
    try:
        df = pd.read_csv(default_path)
        return df
    except Exception as e:
        st.warning("تعذر تحميل الداتا الافتراضية. يمكنك رفع ملف CSV من الشريط الجانبي.")
        return None
def detect_columns(df: pd.DataFrame):
    targets_found = [t for t in TARGETS if t in df.columns]
    feature_cols = [c for c in df.columns if c not in targets_found]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return feature_cols, num_cols, cat_cols, targets_found
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # من غير squared=False
    rmse = mse ** 0.5                          # احسب RMSE يدوي
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
def fmt_metric(m):
    return f"{m:.4f}"
def build_pipeline(num_features, cat_features, random_state=42, n_estimators=400):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    pipe = Pipeline([("prep", pre), ("model", model)])
    return pipe
@st.cache_resource(show_spinner=False)
def train_model(df, target, test_size=0.2, random_state=42):
    feature_cols, num_cols, cat_cols, targets_found = detect_columns(df)
    if target not in targets_found:
        raise ValueError(f"الهدف {target} غير موجود في الأعمدة.")
    X = df[feature_cols].copy()
    y = df[target].copy()
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # خط الأنابيب
    pipe = build_pipeline(num_cols, cat_cols, random_state=random_state)
    # تدريب
    pipe.fit(X_train, y_train)
    # تقييم
    y_pred = pipe.predict(X_test)
    mets = metrics(y_test, y_pred)
    # أهمية الخصائص (Permutation Importance) على العينة
    try:
        # أخذ عينة صغيرة لتسريع الحساب
        subsample = min(1000, X_test.shape[0])
        idx = np.random.RandomState(0).choice(X_test.index, size=subsample, replace=False)
        result = permutation_importance(pipe, X_test.loc[idx], y_test.loc[idx], n_repeats=5, random_state=0, n_jobs=-1)
        # أسماء الميزات بعد الترميز
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)
    except Exception:
        importances = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    return {
        "pipeline": pipe,
        "metrics": mets,
        "X_columns": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_importances": importances
    }
def download_dataframe(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ تحميل CSV", data=csv, file_name=filename, mime="text/csv")
def render_metrics(mets):
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", fmt_metric(mets["MAE"]))
    c2.metric("RMSE", fmt_metric(mets["RMSE"]))
    c3.metric("R²", fmt_metric(mets["R2"]))
def render_feature_form(reference_df: pd.DataFrame, feature_cols, num_cols, cat_cols, key_prefix="single_"):
    inputs = {}
    for col in feature_cols:
        if col in num_cols:
            # قيم افتراضية: الوسط أو 0
            default = float(reference_df[col].dropna().median()) if col in reference_df.columns else 0.0
            val = st.number_input(f"{col}", value=float(default), key=f"{key_prefix}{col}")
        else:
            # اختيار من القيم الملاحظة + خيار كتابة قيمة جديدة
            options = []
            if col in reference_df.columns:
                options = sorted([str(x) for x in reference_df[col].dropna().unique().tolist()][:200])
            choice = st.selectbox(f"{col}", options=options, key=f"{key_prefix}{col}")
            val = choice
        inputs[col] = val
    return pd.DataFrame([inputs])
def header(title, subtitle=None, icon="📊"):
    st.markdown(f"### {icon} {title}")
    if subtitle:
        st.caption(subtitle)
# =======================
# الشريط الجانبي (Sidebar)
# =======================
with st.sidebar:
    st.title("⚙️ الإعدادات")
    st.write("يمكنك رفع ملف CSV بديل أو استخدام الملف الافتراضي.")
    uploaded_csv = st.file_uploader("رفع ملف بيانات (CSV)", type=["csv"], accept_multiple_files=False)
    default_path = "/mnt/data/Expanded_data_with_more_features.csv"
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
    else:
        df = load_data(default_path)
    st.divider()
    st.write("اختياري: تحميل موديل مدرّب (ملف .pkl) لكل هدف")
    up_math = st.file_uploader("MathScore Model (.pkl)", type=["pkl"], key="math_pkl")
    up_read = st.file_uploader("ReadingScore Model (.pkl)", type=["pkl"], key="read_pkl")
    up_write = st.file_uploader("WritingScore Model (.pkl)", type=["pkl"], key="write_pkl")
if df is None or df.empty:
    st.error("لا توجد بيانات. من فضلك ارفع ملف CSV من الشريط الجانبي.")
    st.stop()
# اكتشاف الأعمدة
feature_cols, num_cols, cat_cols, targets_found = detect_columns(df)
# =======================
# الواجهة الرئيسية
# =======================
st.title("منصة التوقع لدرجات الطلاب (Streamlit)")
st.caption("واجهة احترافية للتعامل مع ثلاثة نماذج: MathScore, ReadingScore, WritingScore — تدريب، تقييم، وتنبؤ فردي/جماعي.")
# تبويبات رئيسية
tab_overview, tab_train, tab_predict, tab_bulk = st.tabs(["نظرة عامة", "تدريب/تقييم", "تنبؤ فردي", "تنبؤ جماعي"])
# -----------------------
# نظرة عامة
# -----------------------
with tab_overview:
    header("ملخص البيانات", "حجم البيانات وأنواع الأعمدة", icon="🧾")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد الصفوف", f"{df.shape[0]:,}")
    c2.metric("عدد الأعمدة", f"{df.shape[1]:,}")
    c3.metric("أعمدة عددية", f"{len(num_cols)}")
    c4.metric("أعمدة فئوية", f"{len(cat_cols)}")
    with st.expander("عرض 10 صفوف"):
        st.dataframe(df.sample(10), use_container_width=True)
    with st.expander("القيم المفقودة"):
        na = df.isna().sum().reset_index()
        na.columns = ["column", "missing_count"]
        st.dataframe(na, use_container_width=True)
    st.info("الأهداف المكتشفة في البيانات: " + (", ".join(targets_found) if targets_found else "— لم يتم العثور على الأهداف المتوقعة."))
# -----------------------
# تدريب/تقييم
# -----------------------
with tab_train:
    st.subheader("تدريب وتقييم النماذج")
    st.caption("في حال عدم رفع ملفات .pkl، سيتم تدريب نماذج تلقائيًا باستخدام RandomForest على البيانات الحالية (مع التخزين المؤقت).")
    target_choice = st.multiselect("اختر الأهداف المطلوب تدريبها/تقييمها", TARGETS, default=targets_found if targets_found else TARGETS)
    test_size = st.slider("حجم عينة الاختبار (test_size)", 0.1, 0.4, 0.2, step=0.05)
    if st.button("🚀 تنفيذ التدريب/التقييم"):
        res_store = {}
        for tgt in target_choice:
            if st.session_state.get(f"uploaded_{tgt}"):
                # إذا تم رفع موديل مسبقًا وحُفظ في الجلسة
                res = st.session_state[f"uploaded_{tgt}"]
            elif (tgt == "MathScore" and up_math) or (tgt == "ReadingScore" and up_read) or (tgt == "WritingScore" and up_write):
                # تحميل موديل من pkl والتقييم فقط
                up = up_math if tgt == "MathScore" else up_read if tgt == "ReadingScore" else up_write
                try:
                    pipe = joblib.load(up)
                    X = df[[c for c in feature_cols if c in getattr(pipe, "feature_names_in_", feature_cols)]]
                    y = df[tgt] if tgt in df.columns else None
                    mets = None
                    if y is not None:
                        # تقسيم للتقييم السريع
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
                        pipe.fit(X_tr, y_tr)
                        y_pred = pipe.predict(X_te)
                        mets = metrics(y_te, y_pred)
                    res = {"pipeline": pipe, "metrics": mets, "X_columns": X.columns.tolist(),
                           "num_cols": num_cols, "cat_cols": cat_cols,
                           "feature_importances": pd.DataFrame()}
                    st.session_state[f"uploaded_{tgt}"] = res
                except Exception as e:
                    st.error(f"فشل تحميل موديل {tgt}: {e}")
                    continue
            else:
                res = train_model(df, tgt, test_size=test_size)
                st.session_state[f"uploaded_{tgt}"] = res
            res_store[tgt] = res
        # عرض النتائج
        for tgt, res in res_store.items():
            st.markdown("---")
            header(f"نتائج {tgt}", "مقاييس الأداء وأهمية الخصائص", icon="✅")
            if res["metrics"]:
                render_metrics(res["metrics"])
            else:
                st.info("لا توجد مقاييس متاحة (تم تحميل موديل بدون إعادة تدريب).")
            if not res["feature_importances"].empty:
                with st.expander("عرض أهمية الخصائص (Permutation Importance)"):
                    st.dataframe(res["feature_importances"].head(30), use_container_width=True)
            # زر حفظ الموديل
            try:
                buf = BytesIO()
                joblib.dump(res["pipeline"], buf)
                st.download_button(f"💾 تحميل الموديل ({tgt})", data=buf.getvalue(),
                                   file_name=f"{tgt}_pipeline.pkl", mime="application/octet-stream")
            except Exception:
                pass
# -----------------------
# تنبؤ فردي
# -----------------------
with tab_predict:
    st.subheader("تنبؤ فردي")
    st.caption("املأ المدخلات التالية ثم اختر النموذج المراد استخدامه للحصول على التوقع.")
    st.info("تُبنى المدخلات تلقائيًا من أعمدة البيانات الحاليّة باستثناء الأعمدة الهدف.")
    feature_cols, num_cols, cat_cols, targets_found = detect_columns(df)
    # إنشاء نموذج مرجعي لبناء الفورم (باستخدام قيم الداتا)
    ref_df = df.copy()
    # اختيار النموذج الهدف
    tgt = st.selectbox("اختر الهدف المراد توقعه", TARGETS, index=0)

    # الحصول على Pipeline من الجلسة أو تدريب سريع إذا غير موجود
    if not st.session_state.get(f"uploaded_{tgt}"):
        try:
            st.session_state[f"uploaded_{tgt}"] = train_model(df, tgt)
        except Exception as e:
            st.error(f"لا يمكن تجهيز موديل {tgt}: {e}")
            st.stop()
    pipe = st.session_state[f"uploaded_{tgt}"]["pipeline"]

    with st.form("single_predict_form"):
        single_input_df = render_feature_form(ref_df, feature_cols, num_cols, cat_cols, key_prefix=f"single_{tgt}_")
        submitted = st.form_submit_button("🔮 توقع")
    if submitted:
        try:
            pred = pipe.predict(single_input_df)[0]
            st.success(f"القيمة المتوقعة ({tgt}): **{pred:.2f}**")
            st.json(single_input_df.to_dict(orient="records")[0])
        except Exception as e:
            st.error(f"تعذر إجراء التنبؤ: {e}")
# -----------------------
# تنبؤ جماعي
# -----------------------
with tab_bulk:
    st.subheader("تنبؤ جماعي (Batch)")
    st.caption("ارفع ملف CSV يحتوي على نفس أعمدة الخصائص (بدون أعمدة الهدف إذا رغبت). اختر الهدف ثم احصل على ملف بالتوقعات.")
    bulk_file = st.file_uploader("رفع ملف CSV للتنبؤ الجماعي", type=["csv"], key="bulk_csv")
    tgt_bulk = st.selectbox("اختر الهدف", TARGETS, index=0, key="bulk_target")
    if bulk_file is not None:
        bulk_df = pd.read_csv(bulk_file)
        missing = [c for c in feature_cols if c not in bulk_df.columns]
        if missing:
            st.error("الأعمدة التالية مفقودة في الملف المرفوع: " + ", ".join(missing))
        else:
            # تجهيز الموديل
            if not st.session_state.get(f"uploaded_{tgt_bulk}"):
                try:
                    st.session_state[f"uploaded_{tgt_bulk}"] = train_model(df, tgt_bulk)
                except Exception as e:
                    st.error(f"لا يمكن تجهيز موديل {tgt_bulk}: {e}")
                    st.stop()
            pipe = st.session_state[f"uploaded_{tgt_bulk}"]["pipeline"]

            if st.button("تشغيل التنبؤ الجماعي"):
                try:
                    preds = pipe.predict(bulk_df[feature_cols])
                    out = bulk_df.copy()
                    out[f"{tgt_bulk}_Pred"] = preds
                    st.success("تم حساب التنبؤات بنجاح.")
                    st.dataframe(out.head(20), use_container_width=True)
                    download_dataframe(out, f"{tgt_bulk}_predictions.csv")
                except Exception as e:
                    st.error(f"تعذر إجراء التنبؤ الجماعي: {e}")
# =======================
# Footer
# =======================
st.markdown("""
---
**ملاحظات تقنية**  
- في حال لم ترفع ملفات .pkl، يتم تدريب نماذج RandomForest تلقائيًا مع OneHotEncoder للأعمدة الفئوية.  
- يتم تخزين النتائج مؤقتًا لتسريع الأداء.  
- يدعم التطبيق التنبؤ الفردي والتنبؤ الجماعي مع تصدير النتائج.
""")