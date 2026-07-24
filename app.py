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
    Pandas를 활용해 데이터를 캠페인별/날짜별 피벗 테이블로 요약하여 AI에게 전달합니다.
    """
    cols = [str(c).lower() for c in df.columns]
    
    def find_col(keywords):
        for i, c in enumerate(cols):
            if any(k in c for k in keywords):
                return df.columns[i]
        return None
        
    date_col = find_col(['date', 'day', 'time', '週', '日', '月'])
    campaign_col = find_col(['campaign', 'キャンペーン'])
    cost_col = find_col(['cost', 'spend', '費用', '金額'])
    click_col = find_col(['click', 'クリック'])
    imp_col = find_col(['impression', '表示'])
    conv_col = find_col(['conversion', 'コンバージョン'])
    
    # 콤마 및 퍼센트 제거 후 숫자형 변환
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
        # CPA 및 CTR 계산
        if conv_col and cost_col:
            camp_summary['Calculated_CPA'] = np.where(camp_summary[conv_col] > 0, camp_summary[cost_col] / camp_summary[conv_col], 0)
        if click_col and imp_col:
            camp_summary['Calculated_CTR(%)'] = np.where(camp_summary[imp_col] > 0, (camp_summary[click_col] / camp_summary[imp_col]) * 100, 0)
            
        summary_text += "--- [Campaign Performance Summary (Aggregated)] ---\n"
        summary_text += camp_summary.to_csv(index=False) + "\n\n"
        
    # 2. 날짜별 트렌드 요약 피벗
    if date_col and agg_dict:
        trend_summary = df.groupby(date_col).agg(agg_dict).reset_index().sort_values(by=date_col)
        summary_text += "--- [Time-series Trend Summary (Aggregated)] ---\n"
        summary_text += trend_summary.to_csv(index=False) + "\n"
        
    # 만약 매칭되는 컬럼이 없어 요약에 실패했다면 기본 데이터 제공
    if not summary_text:
        df_cleaned = df.dropna(axis=1, how='all')
        summary_text = df_cleaned.tail(150).to_csv(index=False)
        
    return summary_text

def run_ai_diagnosis(prompt, context, sources_info):
    system_prompt = """
    당신은 'AI ONLABS'의 수석 퍼포먼스 마케터(Senior Performance Marketer)이자 데이터 분석가입니다.
    사용자가 업로드한 광고/비즈니스 데이터의 '피벗 테이블(요약본)'을 바탕으로 날카롭고 실무적인 진단 리포트를 작성해야 합니다.

    [핵심 분석 지침 - CRITICAL INSTRUCTION]
    1. 데이터 구조(행이 몇 개인지 등)에 대한 언급은 절대 금지합니다.
    2. 'Campaign Performance Summary'를 보고 비용은 높으나 전환이 없는(CPA가 비정상적으로 높은) '예산 낭비(Wasted Spend)' 캠페인을 콕 집어내어 예산 축소나 OFF를 권고하세요.
    3. CPA가 낮고 전환 볼륨이 좋은 '스케일업(Scale-up)' 대상 캠페인을 찾아 예산 증액을 제안하세요.
    4. 'Time-series Trend Summary'를 보고 주차별/일자별 예산 소진 트렌드 및 성과 하락 추세를 짚어내세요.
    5. 실무 퍼포먼스 마케팅 용어(예: 매체 효율, 볼륨 최적화, 타겟팅 뎁스, 논타겟 트래픽, 스케일링 등)를 사용하여 전문가처럼 작성하세요.
    6. **모든 출력 결과는 반드시 '한국어(Korean)'로 작성해야 합니다.**

    반드시 아래 4가지 H3(###) 헤딩 구조를 엄격하게 지켜서 마크다운으로 출력하세요:
    ### 1. Executive Summary (현 상황에 대한 마케터 관점의 핵심 요약)
    ### 2. Key Findings (데이터 기반의 주요 발견 사항 및 효율/비효율 캠페인 식별)
    ### 3. Root Causes (수치 변화나 효율 저하의 근본적인 데이터 원인 분석)
    ### 4. Priority Actions (지금 당장 마케터가 광고 매체에서 실행해야 할 구체적인 액션 아이템)
    """
    
    user_message = f"""
    [Current Connected Sources, Coverage, and AGGREGATED DATA]
    {sources_info}
    
    [User Prompt]
    {prompt}
    
    [Additional Context]
    {context if context else "None"}
    
    위 요약 데이터를 바탕으로 실무 퍼포먼스 마케터의 관점에서 진단 리포트를 한국어로 작성해 주세요.
    """
    
    response = llm_client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7
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
                            df = load_csv_smart(file_bytes)
                            coverage = extract_date_range(df)
                            raw_csv_data = prepare_data_for_ai(df) # 피벗 테이블로 정제된 텍스트 반환
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
            context = st.text_input("Additional Context (Optional)", placeholder="e.g., Target CPA is 5000 JPY.")
            
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
