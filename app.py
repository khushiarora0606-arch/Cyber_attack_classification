import html
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.model_evaluation import plot_class_distribution, plot_confusion_matrix, plot_depth_curve, plot_model_comparison, plot_roc_curves, results_table
from src.model_prediction import InputValidationError, ModelNotFoundError, load_model_bundle, predict_frame, summarise_predictions
from src.utils import DEMO_SAMPLE_FILE, EVALUATION_RESULTS, LATEST_UPLOAD_SUMMARY, PREPROCESSING_REPORT, PROCESSED_FILE, SEVERITY, ProjectError, load_json, save_json
st.set_page_config(page_title='Cyber Attack Classification', page_icon='🛡️', layout='wide')
PAGES = ['Overview', 'Attack Classification', 'Model Performance', 'Dataset Analysis', 'About and Ethics']
SEV_COLOR = {'None': '#46C2A5', 'Medium': '#F2B84B', 'High': '#F08A4B', 'Critical': '#EF5B5B'}
CSS = '\n<style>\n.stApp { background: #0B1220; }\n.block-container { padding-top: 2.2rem; max-width: 1250px; }\n.hero { padding: 1.4rem 1.6rem; border: 1px solid #1B2A44; border-radius: 14px; background: #111D33; margin-bottom: 1.2rem; }\n.hero h1 { margin: 0 0 .35rem 0; font-size: 1.9rem; color: #EAF1FF; }\n.hero p { margin: 0; color: #9FB0C8; max-width: 70ch; }\n.card { background: #111B2E; border: 1px solid #1B2A44; border-radius: 12px; padding: 1rem 1.15rem; height: 100%; }\n.card .label { color: #8EA0BB; font-size: .85rem; }\n.card .value { font-size: 1.85rem; font-weight: 600; color: #EAF1FF; }\n.card .sub { color: #7E90AC; font-size: .8rem; }\n.alert { border-radius: 12px; padding: 1rem 1.2rem; margin: .8rem 0 1rem 0; border: 1px solid; }\n.alert .t { font-weight: 600; font-size: 1.15rem; }\n.alert .m { color: #D5DEEE; }\n.pill { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #0B1220; }\n.note { color: #8EA0BB; font-size: .88rem; }\n</style>\n'

@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_model_bundle()

@st.cache_data(show_spinner=False)
def get_results(_mtime):
    return load_json(EVALUATION_RESULTS)

@st.cache_data(show_spinner='Loading dataset...')
def get_processed(_mtime):
    return pd.read_csv(PROCESSED_FILE)

def mtime(path):
    return path.stat().st_mtime if path.exists() else 0.0

def results_or_none():
    return get_results(mtime(EVALUATION_RESULTS))

def cards(items):
    for col, (label, value, *rest) in zip(st.columns(len(items)), items):
        sub = rest[0] if rest else ''
        color = rest[1] if len(rest) > 1 else None
        style = f' style="color:{color}"' if color else ''
        col.markdown(f'<div class="card"><div class="label">{html.escape(str(label))}</div><div class="value"{style}>{html.escape(str(value))}</div><div class="sub">{html.escape(str(sub))}</div></div>', unsafe_allow_html=True)

def alert_banner(alert):
    color, icon = ('#46C2A5', '✅') if alert['level'] == 'normal' else (SEV_COLOR.get(alert['severity'], '#F2B84B'), '🚨')
    st.markdown(f"""<div class="alert" style="border-color:{color}; background:{color}1A;"><div class="t" style="color:{color}">{icon} {html.escape(alert['title'])}</div><div class="m">{html.escape(alert['message'])}</div></div>""", unsafe_allow_html=True)

def show_fig(fig):
    st.pyplot(fig, clear_figure=True, width='stretch')
    plt.close(fig)

def header(title, text):
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>', unsafe_allow_html=True)

def setup_help():
    st.info('**The model has not been trained yet.** Run these commands in the VS Code terminal (see README for details):\n\n```\npython -m src.data_preprocessing\npython -m src.model_training\n```')

def severity_color(s):
    return SEV_COLOR.get(s['highest_severity']) if s['attack_records'] else None

def page_overview():
    header('Cyber Attack Classification and Alerting', 'A machine-learning platform that reads network traffic records, predicts whether each record is normal or one of several attack types, and raises an alert for suspicious traffic. Built on the CICIDS2017 dataset as an academic demonstration.')
    summary = st.session_state.get('last_summary') or load_json(LATEST_UPLOAD_SUMMARY)
    st.subheader('Latest uploaded dataset')
    if not summary:
        st.info('No traffic file has been classified yet. Open **Attack Classification**, upload a CSV and press **Classify traffic** - the numbers below will fill in.')
        cards([('Records analysed', '0'), ('Benign records', '0'), ('Detected attacks', '0'), ('Attack percentage', '0%')])
    else:
        s = summary['summary']
        cards([('Records analysed', f"{s['total_records']:,}", summary.get('source', '')), ('Benign records', f"{s['benign_records']:,}", 'Classified as normal', '#46C2A5'), ('Detected attacks', f"{s['attack_records']:,}", f"Highest severity: {s['highest_severity']}", severity_color(s)), ('Attack percentage', f"{s['attack_percentage']}%", f"Mean confidence {s['mean_confidence']:.1%}")])
        st.write('')
        left, right = st.columns([1.1, 1])
        with left:
            alert_banner(s['alert'])
            st.markdown(f"<span class='note'>File: <b>{html.escape(summary.get('source', 'upload'))}</b> - analysed {html.escape(summary.get('analysed_at', ''))} - {summary.get('n_columns', '?')} columns - model {html.escape(summary.get('model_version', ''))}</span>", unsafe_allow_html=True)
        with right:
            show_fig(plot_class_distribution(s['by_class'], 'Predicted categories in the latest file', dark=True))
    st.subheader('Model in use')
    res = results_or_none()
    if not res:
        setup_help()
    else:
        t = res['test']
        cards([('Best model', res['best_model'], f"Version {res['model_version']}"), ('Test accuracy', f"{t['accuracy']:.2%}", 'Held-out test set'), ('Test macro F1', f"{t['f1_macro']:.3f}", 'Average over all classes'), ('Classes', len(res['classes']), ', '.join(res['classes']))])
    st.markdown("<p class='note'>This application analyses uploaded files only. It does not capture live traffic and does not block attacks.</p>", unsafe_allow_html=True)

def read_upload(uploaded):
    if uploaded.size == 0:
        raise InputValidationError('The uploaded file is empty (0 bytes).')
    try:
        return pd.read_csv(BytesIO(uploaded.getvalue()), low_memory=False, encoding_errors='replace')
    except pd.errors.EmptyDataError:
        raise InputValidationError('The file has no columns or rows. Upload a valid CSV.')
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise InputValidationError(f'The file could not be read as CSV: {str(exc).strip()}')

def page_classification():
    header('Attack Classification', 'Upload a CSV of network flow records (CICIDS2017 format). Each record is classified, scored with a confidence value, and flagged if it looks malicious.')
    try:
        bundle = get_bundle()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        setup_help()
        return
    source = st.radio('Data source', ['Upload a CSV file', 'Use the held-out demo sample'], horizontal=True)
    df, name = (None, '')
    if source == 'Upload a CSV file':
        uploaded = st.file_uploader('Network traffic CSV', type=['csv'], help='Must contain the same flow-feature columns as CICIDS2017.')
        if uploaded is not None:
            name = uploaded.name
            try:
                df = read_upload(uploaded)
            except InputValidationError as exc:
                st.error(str(exc))
                return
    elif DEMO_SAMPLE_FILE.exists():
        df, name = (pd.read_csv(DEMO_SAMPLE_FILE), DEMO_SAMPLE_FILE.name)
        st.caption('About 300 real records taken from the test split. The model never saw them while training.')
    else:
        st.warning('The demo sample is created during training. Run `python -m src.model_training` first.')
        return
    if df is None:
        st.info('Choose a CSV file to begin. Need one? Select the demo sample above, or use any file from the CICIDS2017 download.')
        return
    st.markdown(f'**{name}** - {df.shape[0]:,} rows, {df.shape[1]} columns')
    st.dataframe(df.head(20), width='stretch', height=260)
    if st.button('Classify traffic', type='primary'):
        with st.spinner('Classifying records...'):
            try:
                results, info = predict_frame(bundle, df)
            except ProjectError as exc:
                st.error(str(exc))
                return
        record = {'summary': summarise_predictions(results), 'source': name, 'n_columns': int(df.shape[1]), 'analysed_at': datetime.now().strftime('%Y-%m-%d %H:%M'), 'model_version': bundle['model_version'], 'data_quality': info}
        st.session_state.update(last_summary=record, last_results=results, last_info=info)
        try:
            save_json(record, LATEST_UPLOAD_SUMMARY)
        except OSError:
            pass
    results = st.session_state.get('last_results')
    record = st.session_state.get('last_summary')
    if results is None or record is None or record.get('source') != name:
        return
    s, info = (record['summary'], st.session_state['last_info'])
    alert_banner(s['alert'])
    cards([('Records', f"{s['total_records']:,}"), ('Suspicious records', f"{s['attack_records']:,}", f"{s['attack_percentage']}% of file", severity_color(s) or '#46C2A5'), ('Dominant attack', s['dominant_attack'] or 'None', f"Severity: {s['highest_severity']}"), ('Mean confidence', f"{s['mean_confidence']:.1%}", f"{s['low_confidence_records']:,} below 60%")])
    if info['missing_cells'] or info['infinite_cells']:
        st.caption(f"Data quality: {info['missing_cells']:,} missing/invalid and {info['infinite_cells']:,} infinite cells were replaced using the training medians.")
    if info['unrecognised_labels']:
        st.warning(f"Unrecognised labels in the file were ignored for the accuracy check: {', '.join(info['unrecognised_labels'])}")
    if 'agreement_with_uploaded_labels' in s:
        st.success(f"The file contains a Label column. Predictions match it for {s['agreement_with_uploaded_labels']:.2%} of records.")
    left, right = st.columns(2)
    with left:
        st.markdown('##### Predicted category and severity')
        rows = [{'Category': k, 'Records': v, 'Severity': SEVERITY.get(k, 'Medium')} for k, v in s['by_class'].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    with right:
        show_fig(plot_class_distribution(s['by_class'], 'Classification summary', dark=True))
    st.markdown('##### Record-level results')
    f1, f2 = st.columns(2)
    view = f1.selectbox('Show', ['Suspicious only', 'All records'] + [f'Only {c}' for c in sorted(s['by_class'])])
    min_conf = f2.slider('Minimum confidence', 0.0, 1.0, 0.0, 0.05)
    shown = results
    if view == 'Suspicious only':
        shown = shown[shown['Alert'] == 'ATTACK']
    elif view.startswith('Only '):
        shown = shown[shown['Predicted Class'] == view[5:]]
    shown = shown[shown['Confidence'] >= min_conf]
    st.caption(f'{len(shown):,} of {len(results):,} records match')
    st.dataframe(shown.head(1000), hide_index=True, width='stretch', height=330)
    st.download_button('Download all predictions (CSV)', results.to_csv(index=False).encode('utf-8'), file_name=f"predictions_{name.rsplit('.', 1)[0]}.csv", mime='text/csv')

def page_performance():
    header('Model Performance', 'Four algorithms were tuned with cross-validation and compared on the same held-out test set. All numbers on this page were measured during training.')
    res = results_or_none()
    if not res:
        setup_help()
        return
    table = results_table(res)
    best = res['best_model']
    b = res['models'][best]
    cards([('Best model', best, f"Chosen by {res['selection_metric']}"), ('Test accuracy', f"{b['test']['accuracy']:.2%}"), ('Test macro F1', f"{b['test']['f1_macro']:.3f}", f"CV F1 {b['cv_f1_macro_mean']:.3f} +/- {b['cv_f1_macro_std']:.3f}"), ('Model version', res['model_version'].split('+')[0], f"Trained {res['trained_at']}")])
    st.write('')
    t1, t2, t3, t4 = st.tabs(['Comparison', 'Confusion matrix and ROC', 'Overfitting check', 'Best model details'])
    with t1:
        cols = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'cv_f1_macro', 'fit_seconds']
        show = table[cols].rename(columns={'model': 'Model', 'accuracy': 'Accuracy', 'precision_macro': 'Precision (macro)', 'recall_macro': 'Recall (macro)', 'f1_macro': 'F1 (macro)', 'cv_f1_macro': 'CV F1', 'fit_seconds': 'Tuning time (s)'})
        st.dataframe(show.style.format({c: '{:.4f}' for c in show.columns[1:6]}).highlight_max(subset=list(show.columns[1:6]), color='#1F4A3F'), hide_index=True, width='stretch')
        show_fig(plot_model_comparison(table, dark=True))
        metric = st.selectbox('Compare one metric', ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], format_func=lambda m: m.replace('_', ' ').title())
        st.bar_chart(table.set_index('model')[metric], color='#5B8DEF')
        st.caption('Accuracy alone can look excellent when one class dominates. Macro averages treat every class equally.')
    with t2:
        model = st.selectbox('Model', list(res['models']), index=list(res['models']).index(best))
        r = res['models'][model]
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plot_confusion_matrix(r['test_confusion_matrix'], res['classes'], model, dark=True))
        with c2:
            if r['roc']:
                show_fig(plot_roc_curves(r['roc'], f'{model} - ROC', dark=True))
        st.markdown('##### Per-class precision, recall and F1 (test set)')
        rep = pd.DataFrame(r['test_report']).T.loc[res['classes'], ['precision', 'recall', 'f1-score', 'support']]
        st.dataframe(rep.style.format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:.0f}'}), width='stretch')
        st.caption('False positives: benign traffic flagged as an attack (alert fatigue). False negatives: attacks classified as benign (missed threats).')
    with t3:
        gap = table[['model', 'train_f1_macro', 'validation_f1_macro', 'f1_macro']].copy()
        gap['gap'] = gap['train_f1_macro'] - gap['validation_f1_macro']
        gap.columns = ['Model', 'Train F1', 'Validation F1', 'Test F1', 'Train - validation gap']
        st.dataframe(gap.style.format({c: '{:.4f}' for c in gap.columns[1:]}), hide_index=True, width='stretch')
        st.caption('A large train-validation gap suggests overfitting; low scores on both suggest underfitting.')
        dc = res.get('depth_curve')
        if dc:
            show_fig(plot_depth_curve(dc['depths'], dc['train_f1'], dc['val_f1'], dark=True))
    with t4:
        st.json({'model': best, 'best_hyperparameters': b['best_params'], 'training_rows_used': b['n_train_rows_used'], 'split_sizes': res['split_sizes'], 'confidence_type': res['confidence_type']}, expanded=True)

def page_dataset():
    header('Dataset Analysis', 'A look at the cleaned CICIDS2017 data that the models were trained on.')
    if not PROCESSED_FILE.exists():
        st.warning('The cleaned dataset was not found. Download CICIDS2017, place the CSV files in `data/raw/`, then run `python -m src.data_preprocessing`.')
        return
    df = get_processed(mtime(PROCESSED_FILE))
    report = load_json(PREPROCESSING_REPORT) or {}
    counts = df['Label'].value_counts().to_dict()
    attacks = sum((v for k, v in counts.items() if k != 'BENIGN'))
    cards([('Records (cleaned)', f'{len(df):,}', f"{report.get('rows_read', 0):,} read from raw files" if report else ''), ('Features', df.shape[1] - 1), ('Classes', len(counts)), ('Attack share', f'{attacks / len(df):.1%}', 'After per-class capping')])
    st.write('')
    t1, t2, t3, t4 = st.tabs(['Class distribution', 'Feature statistics', 'Missing values', 'Dataset overview'])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plot_class_distribution(counts, 'Modelling data (after capping)', dark=True))
        with c2:
            before = report.get('class_counts_after_dedup_before_cap')
            if before:
                show_fig(plot_class_distribution(before, 'Before capping (log scale)', dark=True, log=True))
        st.caption('Raw CICIDS2017 is heavily imbalanced (benign traffic dominates). Each class is capped, and models also use balanced class weights.')
        st.bar_chart(pd.Series({'Benign': counts.get('BENIGN', 0), 'Attack': attacks}), color='#5B8DEF')
    with t2:
        feats = [c for c in df.columns if c != 'Label']
        pick = st.multiselect('Features', feats, default=feats[:6])
        if pick:
            st.dataframe(df[pick].describe().T.style.format('{:,.2f}'), width='stretch')
            one = st.selectbox('Histogram of', pick)
            clipped = df[one].clip(df[one].quantile(0.005), df[one].quantile(0.995)).dropna()
            st.bar_chart(pd.cut(clipped, 30).value_counts().sort_index().rename(lambda i: f'{i.mid:,.1f}'), color='#46C2A5')
            st.caption('Values are clipped to the 0.5-99.5 percentile range for display, the same idea used for outlier handling.')
        else:
            st.info('Select at least one feature.')
    with t3:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if report:
            st.markdown(f"Infinite values replaced by NaN during cleaning: **{report.get('inf_replaced', 'n/a'):,}**")
            st.markdown(f"Duplicate rows removed: **{report.get('duplicates_removed_in_chunks', 0) + report.get('duplicates_removed_global', 0):,}**")
        if miss.empty:
            st.success('No missing values remain in the cleaned data.')
        else:
            st.dataframe(pd.DataFrame({'Missing': miss, 'Percent': (miss / len(df) * 100).round(3)}), width='stretch')
            st.caption('Missing values are imputed with the training-set median inside the model pipeline.')
    with t4:
        st.dataframe(df.head(50), width='stretch', height=300)
        info = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str).values, 'Non-null': df.notna().sum().values})
        st.dataframe(info, hide_index=True, width='stretch', height=300)
        if report.get('files'):
            st.markdown('##### Source files')
            st.dataframe(pd.DataFrame(report['files']), hide_index=True, width='stretch')

def page_about():
    header('About and Ethics', 'What this project does, how it was built, and where it should not be trusted.')
    a, b, c = st.tabs(['Project', 'Methodology', 'Ethics and limitations'])
    with a:
        st.markdown('\n**Objective.** Classify network traffic records as BENIGN or one of seven attack categories\n(DDoS, DoS, PortScan, Bot, Brute Force, Web Attack, Other) and raise an alert for malicious traffic.\n\n**Dataset.** CICIDS2017 from the Canadian Institute for Cybersecurity (Kaggle mirror: *Network Intrusion Dataset*).\nIt contains labelled network flows extracted with CICFlowMeter from five days of simulated traffic. The original 15 labels\nare grouped into the categories above.\n\n**Scope.** Offline analysis of uploaded CSV files. This is not a live intrusion detection or prevention system.\n')
    with b:
        st.markdown('\n1. **Clean:** merge CSV files, tidy column names, map labels, replace infinite values, remove duplicates.\n2. **Split:** stratified 70% training, 15% validation, 15% test.\n3. **Pipeline (fitted on training data only):** median imputation, outlier clipping, min-max scaling for KNN and SVM.\n4. **Models:** Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbours, tuned with GridSearchCV cross-validation.\n5. **Selection:** best validation macro-F1; the test set is used only for the final report.\n6. **Deployment:** the saved pipeline serves both this dashboard and a local Flask API.\n\nLinear regression predicts continuous values, so it is not suitable as the main classifier for categorical attack labels.\n')
    with c:
        st.markdown('\n**Privacy.** Flow records can reveal who communicated with whom and when. This project uses a public dataset without IP addresses\nas model inputs. Uploaded files are processed in memory and the app stores only an aggregate summary of the latest upload. In real use,\ntraffic data should be anonymised, access-controlled and kept only as long as needed.\n\n**Fairness and bias.** The model learns only the attacks and the network setup in CICIDS2017 (a 2017 lab environment). New attack types,\nother networks or encrypted/changed protocols may be misclassified. Rare classes (for example Other) have few examples, so their results\nare less reliable. A false alarm can waste analyst time; a missed attack can leave a threat unnoticed.\n\n**Transparency.** Per-class metrics and the simple alert rule are shown so a person can see why traffic was flagged.\nSVM confidence is a softmax of decision scores, not a calibrated probability.\n\n**Security of ML.** Attackers may craft traffic to evade the model (evasion) or poison training data. Treat predictions as decision support\nand combine them with other controls and human review.\n')
        st.markdown('##### Alert severity by category')
        pills = ' '.join((f"""{html.escape(k)} <span class="pill" style="background:{SEV_COLOR.get(v, '#9CA3AF')}">{html.escape(v)}</span>""" for k, v in SEVERITY.items()))
        st.markdown(pills, unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown('### 🛡️ Cyber Attack Classifier')
    page = st.radio('Navigate', PAGES, label_visibility='collapsed')
    st.divider()
    res = results_or_none()
    st.caption(f"Model: **{res['best_model']}**  \nVersion: {res['model_version'].split('+')[0]}" if res else 'Model: not trained yet')
    st.caption('Academic demonstration. Does not monitor live traffic or block attacks.')
{'Overview': page_overview, 'Attack Classification': page_classification, 'Model Performance': page_performance, 'Dataset Analysis': page_dataset, 'About and Ethics': page_about}[page]()
