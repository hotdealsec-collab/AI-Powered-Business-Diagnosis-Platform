import streamlit as st
import pandas as pd
import time
from datetime import datetime
import io
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
# 3. Session State Initialization (프로젝트 격리 구조)
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

    for skip in [2, 1, 0, 3]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), skiprows=skip, encoding=encoding_to_use)
            first_col = str(df.columns[0])
            if len(df.columns) > 2 and 'レポート' not in first_col and 'Report' not in first_col:
                return df
        except Exception:
            continue
            
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding_to_use)

def extract_date_range(df):
    date_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['date', 'day', 'time', '週', '日', '月', '年'])]
    if date_cols:
        date_col = date_cols[0]
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            min_date = parsed_dates.min().strftime('%Y-%m-%d')
            max_date = parsed_dates.max().strftime('%Y-%m-%d')
            return f"{min_date} ~ {max_date}"
        except:
            return "Unknown Coverage"
    return "No Date Column Found"


def prepare_data_for_ai(df):
    """
    Pandas를 활용해 데이터를 캠페인별/날짜별 피벗 테이블로 정확하게 요약합니다.
    """
    cols_lower = [str(c).lower().strip() for c in df.columns]
    
    def find_col(exact_kws, partial_kws):
        # 1. 100% 일치하는 컬럼(정확도 최우선)을 먼저 찾음
        for i, c in enumerate(cols_lower):
            if c in exact_kws: return df.columns[i]
        # 2. 없으면 단어가 포함된 컬럼을 찾음
        for i, c in enumerate(cols_lower):
            if any(k in c for k in partial_kws): return df.columns[i]
        return None
        
    date_col = find_col(['週', '日', 'date', 'day'], ['time', 'date', '月'])
    # 캠페인 이름 정확히 매칭 ('キャンペーン タイプ' 등과 혼동 방지)
    campaign_col = find_col(['キャンペーン', 'campaign'], ['campaign', 'キャンペーン'])
    cost_col = find_col(['費用', 'cost', 'spend'], ['金額', 'cost'])
    click_col = find_col(['クリック数', 'clicks'], ['click', 'クリック'])
    imp_col = find_col(['表示回数', 'impressions'], ['imp', '表示'])
    conv_col = find_col(['コンバージョン', 'conversions'], ['conv', 'コンバージョン'])
    
    # 콤마 및 퍼센트 제거 후 숫자형(Float) 변환
    for col in [cost_col, click_col, imp_col, conv_col]:
        if col and col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    summary_text = ""
    agg_dict = {}
    if cost_col: agg_dict[cost_col] = 'sum'
    if click_col: agg_dict[click_col] = 'sum'
    if imp_col: agg_dict[imp_col] = 'sum'
    if conv_col: agg_dict[conv_col] = 'sum'
    
    # 1. 캠페인별 성과 요약 피벗
    if campaign_col and agg_dict:
        camp_summary = df.groupby(campaign_col).agg(agg_dict).reset_index()
        # CPA 및 CTR 계산 (ZeroDivision 방지)
        if conv_col and cost_col:
            camp_summary['CPA'] = np.where(camp_summary[conv_col] > 0, (camp_summary[cost_col] / camp_summary[conv_col]).round(0), 0)
        if click_col and imp_col:
            camp_summary['CTR(%)'] = np.where(camp_summary[imp_col] > 0, ((camp_summary[click_col] / camp_summary[imp_col]) * 100).round(2), 0)
            
        summary_text += "--- [Campaign Performance Summary (Aggregated)] ---\n"
        summary_text += camp_summary.to_csv(index=False) + "\n\n"
        
    # 2. 날짜별 트렌드 요약 피벗
    if date_col and agg_dict:
        trend_summary = df.groupby(date_col).agg(agg_dict).reset_index().sort_values(by=date_col)
        if conv_col and cost_col:
            trend_summary['CPA'] = np.where(trend_summary[conv_col] > 0, (trend_summary[cost_col] / trend_summary[conv_col]).round(0), 0)
        summary_text += "--- [Time-series Trend Summary (Aggregated)] ---\n"
        summary_text += trend_summary.to_csv(index=False) + "\n"
        
    if not summary_text:
        df_cleaned = df.dropna(axis=1, how='all')
        summary_text = df_cleaned.tail(150).to_csv(index=False)
        
    return summary_text

def run_ai_diagnosis(prompt, context, sources_info):
    system_prompt = """
    당신은 7년 차 탑티어 퍼포먼스 마케터(Performance Marketer)입니다.
    제공된 데이터(피벗 테이블)를 분석하여 광고주가 즉시 광고 시스템에 접속해 조치를 취할 수 있는 수준의 '매체 최적화 액션플랜'을 도출해야 합니다.

    [핵심 분석 룰]
    1. 'Campaign Performance Summary'의 캠페인 이름(예: 【KP】ACe_Demandgen 등)을 반드시 정확하게 명시하여 분석하세요.
    2. CPA가 극단적으로 높거나, 비용(Cost)은 많이 썼는데 전환(Conversion)이 0인 캠페인은 "즉시 OFF 또는 예산 대폭 축소" 대상입니다.
    3. CPA가 타겟 단가에 부합하거나 효율이 매우 좋은 캠페인은 "예산 증액(Scale-up)" 대상입니다.
    4. 분석할 때 반드시 실제 수치(비용, CPA, 전환수 등)를 괄호나 문장 속에 포함하여 설득력을 높이세요.
    5. 'Time-series Trend'를 보고 주차별/일자별로 매체 효율이 악화되고 있는지, 개선되고 있는지 짚어내세요.
    6. **반드시 마케팅 실무 용어(예: 지면 최적화, 머신러닝 안정화, 디마케팅, ROAS/CPA 한계점, 타겟팅 뎁스 등)를 자연스럽게 구사하세요.**

    반드시 아래 4가지 H3(###) 헤딩 구조를 엄격하게 지켜서 마크다운으로 출력하세요. 절대 영어로 출력하지 마세요:
    ### 1. Executive Summary (성과 현황 요약)
    ### 2. Key Findings (캠페인별 세부 효율 진단)
    ### 3. Root Causes (효율 상승/하락의 데이터적 근거)
    ### 4. Priority Actions (마케터가 지금 당장 실행해야 할 매체 최적화 액션)
    """
    
    user_message = f"""
    [데이터 요약본 (캠페인별 & 시계열)]
    {sources_info}
    
    [사용자 분석 요청]
    {prompt}
    
    [추가 비즈니스 컨텍스트]
    {context if context else "None"}
    
    위 데이터를 바탕으로 실무 퍼포먼스 마케터의 관점에서 진단 리포트를 한국어로 작성해 주세요.
    """
    
    response = llm_client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content


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
