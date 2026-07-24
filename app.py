import streamlit as st
import pandas as pd
import time
from datetime import datetime
import io
from supabase import create_client, Client
from openai import OpenAI

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
    st.session_state.projects = [{"name": "AJIOKA", "source_count": 0, "last_analysis": "Never"}]
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'archives' not in st.session_state:
    st.session_state.archives = []
    
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
    """Google Ads 등 메타데이터가 포함된 CSV를 자동으로 인식하고 읽어옵니다."""
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
    """AI가 분석할 수 있도록 핵심 데이터를 텍스트로 변환합니다."""
    # 빈 컬럼 제거 및 최근 150줄로 제한 (토큰 한도 방지)
    df_cleaned = df.dropna(axis=1, how='all')
    df_sample = df_cleaned.tail(150)
    return df_sample.to_csv(index=False)

def run_ai_diagnosis(prompt, context, sources_info):
    """OpenAI API를 호출하여 진단 리포트를 생성합니다."""
    system_prompt = """
    You are an expert AI Business & Performance Marketing Consultant for 'AI ONLABS'.
    You will be provided with actual business data (in CSV format) uploaded by the user.
    
    CRITICAL INSTRUCTION:
    - Do NOT talk about the file structure (e.g., "The dataset has 95 rows and 29 columns").
    - Do NOT suggest "getting more data" as a primary action.
    - Instead, deeply analyze the actual business metrics in the data (e.g., Cost, Clicks, Impressions, CPA, Conversions, ROAS, trends over time, campaign performance).
    - Identify specific campaigns that are working well, and which ones are failing or spending inefficiently. Provide data-backed reasons.
    
    You MUST format your response strictly using these 4 exact headings (use Markdown H3 ###):
    ### 1. Executive Summary
    ### 2. Key Findings
    ### 3. Root Causes
    ### 4. Priority Actions
    Do not add extra sections. Keep it professional, highly analytical, and actionable.
    """
    
    user_message = f"""
    [Current Connected Sources, Coverage, and RAW DATA]
    {sources_info}
    
    [User Prompt]
    {prompt}
    
    [Additional Context]
    {context if context else "None"}
    
    Based on the provided raw data timeline and context, please provide the diagnosis. Focus on the actual metric numbers.
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
    for idx, proj in enumerate(st.session_state.projects):
        col = cols[idx % 3]
        with col:
            st.markdown(f'''
            <div class="card">
                <h3 style="margin-top:0;">{proj["name"]}</h3>
                <p>{proj["source_count"]} Connected Sources</p>
                <p class="coverage-text">Last Analysis: {proj["last_analysis"]}</p>
            </div>
            ''', unsafe_allow_html=True)
            if st.button("Open Project →", key=f"btn_proj_{idx}", use_container_width=True):
                st.session_state.current_project = proj["name"]
                navigate("Workspace")
                
    st.divider()
    st.subheader("+ Create New Project")
    with st.form("new_project_form", clear_on_submit=True):
        new_proj_name = st.text_input("Project Name")
        if st.form_submit_button("Create Project") and new_proj_name:
            st.session_state.projects.append({"name": new_proj_name, "source_count": 0, "last_analysis": "Never"})
            st.rerun()

# ==========================================
# 6. View: Workspace
# ==========================================
def view_workspace():
    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {st.session_state.current_project}"): navigate("Projects")
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
                            raw_csv_data = prepare_data_for_ai(df) # AI용 데이터 추출
                        except Exception as e:
                            st.error(f"데이터 파싱 에러: {e}")
                            st.stop()
                        
                        file_name = f"{st.session_state.current_project}_{platform}_{int(time.time())}.csv"
                        try:
                            supabase.storage.from_(BUCKET_NAME).upload(file_name, file_bytes)
                            upload_success = True
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                            upload_success = False

                        if upload_success:
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                            st.session_state.sources.append({
                                "name": platform, "status": "🟢 Ready", "latest": now_str,
                                "coverage": coverage, "history": [f"{now_str} (Initial)"], 
                                "raw_data": raw_csv_data # 메모리에 실데이터 적재
                            })
                            st.session_state.projects[0]["source_count"] = len(st.session_state.sources)
                            st.session_state.show_add_source = False
                            st.rerun()
                    
        st.write("") 
        for idx, src in enumerate(st.session_state.sources):
            with st.expander(f"{src['name']}  ({src['status'].split(' ')[0]})", expanded=(idx==0)):
                st.markdown(f"**Status:** {src['status']}<br>**Latest:** {src['latest']}<br>**Coverage:** `{src['coverage']}`", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**▶ Data History (Timeline)**")
                for hist in src['history']: st.caption(f"📄 dataset.csv - {hist}")

    with right_col:
        st.subheader("Analysis")
        source_names = ", ".join([s['name'] for s in st.session_state.sources])
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
                if not st.session_state.sources:
                    st.error("Please add at least one data source first!")
                elif not prompt:
                    st.warning("Please enter a prompt.")
                else:
                    with st.spinner("AI is analyzing real business metrics..."):
                        
                        # 각 소스의 '실제 데이터'를 프롬프트에 조립
                        sources_info = ""
                        for s in st.session_state.sources:
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
                        st.session_state.archives.insert(0, st.session_state.analysis_result)
                        st.session_state.projects[0]["last_analysis"] = datetime.now().strftime("%b %d")
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
    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {st.session_state.current_project}"): navigate("Projects")
    with nav_col2:
        if st.button("Workspace", use_container_width=True): navigate("Workspace")
    with nav_col3: st.button("Archive", use_container_width=True, disabled=True)
        
    st.divider()
    left_col, right_col = st.columns([3.5, 6.5], gap="large")
    
    with left_col:
        st.subheader("Diagnosis History")
        for idx, arch in enumerate(st.session_state.archives):
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
