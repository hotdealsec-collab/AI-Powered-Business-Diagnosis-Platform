import streamlit as st
import pandas as pd
import time
from datetime import datetime
import io
import re
from supabase import create_client, Client
from openai import OpenAI
import numpy as np

# ==========================================
# 1. API & DB Setup
# ==========================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error("⚠️ Secrets 설정이 누락되었거나 잘못되었습니다. Streamlit Cloud의 Settings > Secrets를 확인해주세요.")
    st.stop()

BUCKET_NAME = "datasets-ai-powered-business-diagnosis-platform"

# ==========================================
# 2. Page Configuration & CSS
# ==========================================
st.set_page_config(page_title="AI ONLABS MVP", layout="wide", initial_sidebar_state="collapsed")

st.markdown('''
<style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .card {
        border: 1px solid #E5E7EB; border-radius: 8px; padding: 24px; margin-bottom: 16px;
        background-color: #FFFFFF; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .status-ready { color: #10B981; font-weight: 600; }
    .coverage-text { font-size: 0.9em; color: #6B7280; }
    footer {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

# ==========================================
# 3. Session State Initialization
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'Projects'
if 'current_project' not in st.session_state: st.session_state.current_project = None

if 'projects' not in st.session_state:
    st.session_state.projects = {
        "AJIOKA": {
            "source_count": 0, 
            "last_analysis": "Never", 
            "sources": [], 
            "archives": []
        }
    }
    
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
if 'prompt_input' not in st.session_state: st.session_state.prompt_input = ""
if 'show_add_source' not in st.session_state: st.session_state.show_add_source = False

def navigate(page_name):
    st.session_state.page = page_name
    st.rerun()

# ==========================================
# 4. Helper Functions (Data & AI)
# ==========================================
def load_csv_smart(file_bytes):
    encoding_to_use = 'utf-8-sig'
    for enc in ['utf-8-sig', 'utf-8', 'shift_jis', 'cp932', 'utf-16']:
        try:
            file_bytes.decode(enc)
            encoding_to_use = enc
            break
        except Exception:
            continue

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding_to_use, comment='#')
        if len(df.columns) > 2:
            return df, encoding_to_use
    except Exception:
        pass

    for skip in range(13):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), skiprows=skip, encoding=encoding_to_use)
            first_col = str(df.columns[0])
            if len(df.columns) > 2 and 'レポート' not in first_col and 'Report' not in first_col:
                return df, encoding_to_use
        except Exception:
            continue
            
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding_to_use), encoding_to_use

def extract_date_range(df, file_bytes, encoding):
    date_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['date', 'day', 'time', '週', '日', '月', '年'])]
    if date_cols:
        date_col = date_cols[0]
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            min_date = parsed_dates.min().strftime('%Y-%m-%d')
            max_date = parsed_dates.max().strftime('%Y-%m-%d')
            return f"{min_date} ~ {max_date}"
        except:
            pass
            
    try:
        decoded_text = file_bytes.decode(encoding)
        start_match = re.search(r'開始日:\s*(\d{8})', decoded_text)
        end_match = re.search(r'終了日:\s*(\d{8})', decoded_text)
        if start_match and end_match:
            start_date = pd.to_datetime(start_match.group(1)).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(end_match.group(1)).strftime('%Y-%m-%d')
            return f"{start_date} ~ {end_date}"
    except Exception:
        pass
        
    return "Unknown Coverage (Summary)"

def prepare_data_for_ai(df):
    cols_lower = [str(c).lower().strip() for c in df.columns]
    
    def find_col(exact_kws, partial_kws):
        for i, c in enumerate(cols_lower):
            if c in exact_kws: return df.columns[i]
        for i, c in enumerate(cols_lower):
            if any(k in c for k in partial_kws): return df.columns[i]
        return None
        
    date_col = find_col(['週', '日', 'date', 'day'], ['time', 'date', '月'])
    campaign_col = find_col(['キャンペーン', 'campaign'], ['campaign', 'キャンペーン'])
    cost_col = find_col(['費用', '広告費用', 'cost', 'spend'], ['金額', 'cost'])
    click_col = find_col(['クリック数', '広告のクリック数', 'clicks'], ['click', 'クリック'])
    imp_col = find_col(['表示回数', '広告の表示回数', 'impressions'], ['imp', '表示'])
    conv_col = find_col(['コンバージョン', 'キーイベント', 'conversions'], ['conv', 'コンバージョン', 'キーイベント'])
    rev_col = find_col(['合計収益', 'revenue', 'value', '購入値'], ['収益', 'revenue', 'value'])
    
    for col in [cost_col, click_col, imp_col, conv_col, rev_col]:
        if col and col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    summary_text = ""
    agg_dict = {}
    if cost_col: agg_dict[cost_col] = 'sum'
    if click_col: agg_dict[click_col] = 'sum'
    if imp_col: agg_dict[imp_col] = 'sum'
    if conv_col: agg_dict[conv_col] = 'sum'
    if rev_col: agg_dict[rev_col] = 'sum'
    
    if campaign_col and agg_dict:
        camp_summary = df.groupby(campaign_col).agg(agg_dict).reset_index()
        if conv_col and cost_col:
            camp_summary['CPA'] = np.where(camp_summary[conv_col] > 0, (camp_summary[cost_col] / camp_summary[conv_col]).round(0), 0)
        if click_col and imp_col:
            camp_summary['CTR(%)'] = np.where(camp_summary[imp_col] > 0, ((camp_summary[click_col] / camp_summary[imp_col]) * 100).round(2), 0)
        if conv_col and click_col:
            camp_summary['CVR(%)'] = np.where(camp_summary[click_col] > 0, ((camp_summary[conv_col] / camp_summary[click_col]) * 100).round(2), 0)
        if rev_col and cost_col:
            camp_summary['ROAS(%)'] = np.where(camp_summary[cost_col] > 0, ((camp_summary[rev_col] / camp_summary[cost_col]) * 100).round(0), 0)
            
        summary_text += "--- [Campaign Performance Summary (Aggregated)] ---\n"
        summary_text += camp_summary.to_csv(index=False) + "\n\n"
        
    if date_col and agg_dict:
        trend_summary = df.groupby(date_col).agg(agg_dict).reset_index().sort_values(by=date_col)
        if conv_col and cost_col:
            trend_summary['CPA'] = np.where(trend_summary[conv_col] > 0, (trend_summary[cost_col] / trend_summary[conv_col]).round(0), 0)
        if conv_col and click_col:
            trend_summary['CVR(%)'] = np.where(trend_summary[click_col] > 0, ((trend_summary[conv_col] / trend_summary[click_col]) * 100).round(2), 0)
        if rev_col and cost_col:
            trend_summary['ROAS(%)'] = np.where(trend_summary[cost_col] > 0, ((trend_summary[rev_col] / trend_summary[cost_col]) * 100).round(0), 0)
        summary_text += "--- [Time-series Trend Summary (Aggregated)] ---\n"
        summary_text += trend_summary.to_csv(index=False) + "\n"
        
    if not summary_text:
        df_cleaned = df.dropna(axis=1, how='all')
        summary_text = df_cleaned.tail(150).to_csv(index=False)
        
    return summary_text

def run_ai_diagnosis(prompt, context, sources_info):
    system_prompt = """
    당신은 7년 차 탑티어 퍼포먼스 마케터(Performance Marketer)입니다.
    제공된 데이터(피벗 테이블)를 분석하여 광고주가 즉시 실행할 수 있는 '매체 최적화 액션플랜'을 도출해야 합니다.

    [🚨 치명적 주의사항 - CPA 수학적 논리 (절대 틀리지 마세요)]
    사용자가 제시한 목표 CPA(Target CPA)와 각 캠페인의 실제 CPA를 비교할 때, 반드시 아래 사고 과정을 거치세요:
    1. 사용자의 타겟 CPA 숫자를 확인합니다. (예: 3000)
    2. 캠페인의 실제 CPA 숫자를 확인합니다. (예: 139)
    3. 139는 3000보다 작으므로 "목표 달성(매우 우수함, Scale-up 대상)"으로 판정합니다.
    4. 반대로 4500은 3000보다 크므로 "목표 초과(비효율, OFF 대상)"로 판정합니다.
    절대 숫자의 크기 비교를 반대로 해석하여 효율/비효율을 거꾸로 말하지 마세요.

    [핵심 퍼포먼스 마케팅 진단 로직]
    1. **절대 룰 (ROAS):** ROAS(광고비용 대비 수익률)는 100%를 기준으로 높을수록 우수합니다. 
    2. **퍼넬 진단 (CTR vs CVR 분석):**
       - [CTR은 높은데 CVR이 낮음]: 광고 소재(Creative)의 후킹은 좋으나 랜딩 페이지 경쟁력이 떨어지거나 과대광고일 확률이 높음.
       - [CTR은 낮으나 CVR이 높음]: 타겟팅이 너무 좁거나 소재 매력도는 떨어지지만 상품 자체의 소구력은 좋음.
    3. **예산 낭비(Wasted Spend) 색출:** 모수가 충분함에도 CPA가 타겟 CPA를 심각하게 초과하거나 전환이 0인 캠페인은 "즉시 OFF"를 권고하세요.
    
    [작성 규칙]
    - 캠페인 이름을 정확히 명시하고, 분석 시 반드시 괄호 안에 실제 수치(CPA, ROAS, CVR 등)를 넣어 근거를 대세요.
    - 한국어(Korean)로만 출력하고, 아래 4가지 H3(###) 마크다운 헤딩 구조를 반드시 지키세요.

    ### 1. Executive Summary (성과 현황 요약)
    ### 2. Key Findings (캠페인별 세부 효율 진단 - 타겟 CPA 달성 여부 반드시 포함)
    ### 3. Root Causes (효율 상승/하락의 데이터적 원인 진단)
    ### 4. Priority Actions (마케터가 당장 실행해야 할 예산 최적화 액션)
    """
    
    user_message = f"""
    [데이터 요약본 (캠페인별 & 시계열)]
    {sources_info}
    
    [사용자 분석 요청]
    {prompt}
    
    [추가 비즈니스 컨텍스트]
    {context if context else "None"}
    
    위 데이터를 바탕으로 실무 퍼포먼스 마케터의 관점에서 진단 리포트를 작성해 주세요.
    """
    
    # Rate Limit 방지를 위해 gpt-4o-mini 사용, 수학적 논리 강화를 위해 temperature 0.2 적용
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2 
    )
    return response.choices[0].message.content

# ==========================================
# 5. View: Projects
# ==========================================
def view_projects():
    st.title("AI ONLABS")
    st.write("Select a project to enter the workspace.")
    st.divider()
    
    cols = st.columns(3)
    idx = 0
    for proj_name, proj_data in st.session_state.projects.items():
        col = cols[idx % 3]
        with col:
            st.markdown(f'''
            <div class="card">
                <h3 style="margin-top:0;">{proj_name}</h3>
                <p>{proj_data["source_count"]} Connected Sources</p>
                <p class="coverage-text">Last Analysis: {proj_data["last_analysis"]}</p>
            </div>
            ''', unsafe_allow_html=True)
            if st.button("Open Project →", key=f"btn_proj_{idx}", use_container_width=True):
                st.session_state.current_project = proj_name
                navigate("Workspace")
        idx += 1
                
    st.divider()
    st.subheader("+ Create New Project")
    with st.form("new_project_form", clear_on_submit=True):
        new_proj_name = st.text_input("Project Name")
        if st.form_submit_button("Create Project") and new_proj_name:
            if new_proj_name not in st.session_state.projects:
                st.session_state.projects[new_proj_name] = {
                    "source_count": 0, 
                    "last_analysis": "Never", 
                    "sources": [], 
                    "archives": []
                }
            st.rerun()

# ==========================================
# 6. View: Workspace
# ==========================================
def view_workspace():
    curr_proj = st.session_state.current_project
    proj_data = st.session_state.projects[curr_proj]
    
    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {curr_proj}"): navigate("Projects")
    with nav_col2: st.button("Workspace", use_container_width=True, disabled=True)
    with nav_col3:
        if st.button("Archive", use_container_width=True): navigate("Archive")
        
    st.divider()
    left_col, right_col = st.columns([3.5, 6.5], gap="large")
    
    with left_col:
        st.subheader("Sources")
        if st.button("+ Add Source", use_container_width=True, type="secondary"):
            st.session_state.show_add_source = not st.session_state.show_add_source
            
        if st.session_state.show_add_source:
            with st.container(border=True):
                platform = st.selectbox("Platform", ["Google Ads", "Meta Ads", "GA4", "Shopify"])
                uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
                if st.button("Connect & Upload") and uploaded_file:
                    with st.spinner("Uploading to Supabase & Analyzing timeline..."):
                        file_bytes = uploaded_file.getvalue()
                        try:
                            df, enc = load_csv_smart(file_bytes)
                            coverage = extract_date_range(df, file_bytes, enc)
                            raw_csv_data = prepare_data_for_ai(df)
                        except Exception as e:
                            st.error(f"데이터 파싱 에러: {e}")
                            st.stop()
                        
                        file_name = f"{curr_proj}_{platform}_{int(time.time())}.csv"
                        try:
                            supabase.storage.from_(BUCKET_NAME).upload(file_name, file_bytes)
                            upload_success = True
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                            upload_success = False

                        if upload_success:
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                            proj_data["sources"].append({
                                "name": platform, "status": "🟢 Ready", "latest": now_str,
                                "coverage": coverage, "history": [f"{now_str} (Initial)"], 
                                "raw_data": raw_csv_data
                            })
                            proj_data["source_count"] = len(proj_data["sources"])
                            st.session_state.show_add_source = False
                            st.rerun()
                    
        st.write("") 
        for idx, src in enumerate(proj_data["sources"]):
            with st.expander(f"{src['name']}  ({src['status'].split(' ')[0]})", expanded=(idx==0)):
                st.markdown(f"**Status:** {src['status']}<br>**Latest:** {src['latest']}<br>**Coverage:** `{src['coverage']}`", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**▶ Data History (Timeline)**")
                for hist in src['history']: st.caption(f"📄 dataset.csv - {hist}")

    with right_col:
        st.subheader("Analysis")
        source_names = ", ".join([s['name'] for s in proj_data["sources"]])
        st.info(f"💡 **Current Timeline Context**\n\nSources: {source_names if source_names else 'None'}")
        
        if st.session_state.analysis_result is None:
            st.markdown("### What would you like to analyze?")
            s_col1, s_col2, s_col3 = st.columns(3)
            if s_col1.button("Performance Review", use_container_width=True): st.session_state.prompt_input = "Performance Review"
            if s_col2.button("Campaign Diagnosis", use_container_width=True): st.session_state.prompt_input = "Campaign Diagnosis"
            if s_col3.button("Trend Analysis", use_container_width=True): st.session_state.prompt_input = "Trend Analysis"
            
            prompt = st.text_area("Prompt", value=st.session_state.prompt_input, height=100)
            context = st.text_input("Additional Context (Optional)", placeholder="e.g., Target CPA is 3000 JPY.")
            
            if st.button("Run Diagnosis", type="primary"):
                if not proj_data["sources"]:
                    st.error("Please add at least one data source first!")
                elif not prompt:
                    st.warning("Please enter a prompt.")
                else:
                    with st.spinner("AI is analyzing real business metrics..."):
                        sources_info = ""
                        for s in proj_data["sources"]:
                            sources_info += f"\n--- Source: {s['name']} (Coverage: {s['coverage']}) ---\n"
                            sources_info += f"```csv\n{s['raw_data']}\n```\n"
                        
                        ai_report = run_ai_diagnosis(prompt, context, sources_info)
                        
                        st.session_state.analysis_result = {
                            "title": prompt,
                            "date": datetime.now().strftime("%b %d"),
                            "sources": source_names,
                            "coverage": "Evaluated on all available timelines",
                            "report_content": ai_report
                        }
                        proj_data["archives"].insert(0, st.session_state.analysis_result)
                        proj_data["last_analysis"] = datetime.now().strftime("%b %d")
                        st.rerun()
        else:
            res = st.session_state.analysis_result
            if st.button("← Back to Prompt"):
                st.session_state.analysis_result = None
                st.rerun()
                
            st.markdown(f"# {res['title']}")
            st.caption(f"Generated: {res['date']} | Sources: {res['sources']}")
            st.divider()
            st.markdown(res['report_content'])

# ==========================================
# 7. View: Archive & Router Logic
# ==========================================
def view_archive():
    curr_proj = st.session_state.current_project
    proj_data = st.session_state.projects[curr_proj]

    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {curr_proj}"): navigate("Projects")
    with nav_col2:
        if st.button("Workspace", use_container_width=True): navigate("Workspace")
    with nav_col3: st.button("Archive", use_container_width=True, disabled=True)
        
    st.divider()
    left_col, right_col = st.columns([3.5, 6.5], gap="large")
    
    with left_col:
        st.subheader("Diagnosis History")
        for idx, arch in enumerate(proj_data["archives"]):
            if st.button(f"📄 {arch['title']}\n\n{arch['date']}", key=f"arch_{idx}", use_container_width=True):
                st.session_state.selected_archive = arch
                
    with right_col:
        if 'selected_archive' in st.session_state:
            arch = st.session_state.selected_archive
            st.markdown(f"# {arch['title']}")
            st.caption(f"Generated: {arch['date']} | Sources: {arch['sources']}")
            st.divider()
            st.markdown(arch['report_content'])
            st.divider()
            if st.button("Re-run with latest timeline data"):
                st.session_state.prompt_input = arch['title']
                navigate("Workspace")
        else:
            st.info("Select a diagnosis from the history to view its details.")

if st.session_state.page == 'Projects': view_projects()
elif st.session_state.page == 'Workspace': view_workspace()
elif st.session_state.page == 'Archive': view_archive()
