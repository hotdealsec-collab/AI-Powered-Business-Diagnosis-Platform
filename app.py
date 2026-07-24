
import streamlit as st
import time

# ==========================================
# 1. Page Configuration & CSS
# ==========================================
st.set_page_config(page_title="AI ONLABS MVP", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for NotebookLM-like minimalist feel
st.markdown('''
<style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .status-ready { color: #10B981; font-weight: 600; }
    .status-review { color: #F59E0B; font-weight: 600; }
    .coverage-text { font-size: 0.9em; color: #6B7280; }
    
    /* Hide Streamlit default UI elements as much as possible */
    footer {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'Projects'

if 'current_project' not in st.session_state:
    st.session_state.current_project = None

if 'sources' not in st.session_state:
    st.session_state.sources = [
        {"name": "Google Ads", "status": "🟢 Ready", "latest": "2026-07-24 09:35", "coverage": "2026-01-03 ~ 2026-07-24", "history": ["2026-07-24 09:35 (v4)", "2026-07-23 18:10 (v3)"]},
        {"name": "GA4", "status": "🟡 Needs Review", "latest": "2026-07-20 14:00", "coverage": "2026-01-03 ~ 2026-07-20", "history": ["2026-07-20 14:00 (v1)"]}
    ]

if 'archives' not in st.session_state:
    st.session_state.archives = [
        {"title": "Campaign Diagnosis", "date": "Jul 21", "coverage": "2026-01-03 ~ 2026-07-20", "sources": "Google Ads", "prompt": "Diagnose recent campaign drops"}
    ]
    
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
    
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = ""

# ==========================================
# 3. Navigation Helper
# ==========================================
def navigate(page_name):
    st.session_state.page = page_name
    st.rerun()

# ==========================================
# 4. View: Projects
# ==========================================
def view_projects():
    st.title("AI ONLABS")
    st.write("Select a project to enter the workspace.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="card">
            <h3 style="margin-top:0;">AJIOKA</h3>
            <p>3 Connected Sources</p>
            <p class="coverage-text">Last Analysis: Jul 24</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Open Project →", key="btn_aji", use_container_width=True):
            st.session_state.current_project = "AJIOKA"
            navigate("Workspace")
            
    with col2:
        st.markdown('''
        <div class="card">
            <h3 style="margin-top:0;">THE TOKYO PASS</h3>
            <p>1 Connected Source</p>
            <p class="coverage-text">Last Analysis: Jul 20</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Open Project →", key="btn_ttp", use_container_width=True):
            st.session_state.current_project = "THE TOKYO PASS"
            navigate("Workspace")
            
    with col3:
        st.markdown('''
        <div class="card" style="display: flex; align-items: center; justify-content: center; height: 135px; border-style: dashed;">
            <h3 style="color: #9CA3AF; margin: 0;">+ New Project</h3>
        </div>
        ''', unsafe_allow_html=True)

# ==========================================
# 5. View: Workspace
# ==========================================
def view_workspace():
    # Top Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {st.session_state.current_project}"):
            navigate("Projects")
    with nav_col2:
        st.button("Workspace", use_container_width=True, disabled=True)
    with nav_col3:
        if st.button("Archive", use_container_width=True): 
            navigate("Archive")
        
    st.divider()
    
    # 2-Column Layout (35% / 65%)
    left_col, right_col = st.columns([3.5, 6.5], gap="large")
    
    # --- LEFT PANEL: Sources ---
    with left_col:
        st.subheader("Sources")
        if st.button("+ Add Source", use_container_width=True, type="secondary"):
            st.info("Modal UI: Select platform (Advertising, Analytics, CRM) -> Upload Dataset")
            
        st.write("") # spacing
            
        for idx, src in enumerate(st.session_state.sources):
            with st.expander(f"{src['name']}  ({src['status'].split(' ')[0]})", expanded=(idx==0)):
                st.markdown(f"**Status:** {src['status']}")
                st.markdown(f"**Latest Update:** {src['latest']}")
                st.markdown(f"**Coverage:** `{src['coverage']}`")
                
                st.markdown("---")
                st.markdown("**▶ Data History**")
                for hist in src['history']:
                    st.caption(f"📄 dataset_{src['name'].lower().replace(' ', '_')}.csv - {hist}")
                
                st.write("")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Add Dataset", key=f"add_{idx}", use_container_width=True):
                        st.success("New Date Range Detected. Appending to timeline...")
                with col_b:
                    st.button("Replace Latest", key=f"rep_{idx}", use_container_width=True)

    # --- RIGHT PANEL: Analysis ---
    with right_col:
        st.subheader("Analysis")
        
        # Context Banner
        st.info("💡 **Current Timeline Context**\n\nSources: Google Ads, GA4 | Coverage: `2026-01-03 ~ 2026-07-24`")
        
        # If no result is present, show Prompt input
        if st.session_state.analysis_result is None:
            st.markdown("### What would you like to analyze?")
            
            # Suggested chips
            s_col1, s_col2, s_col3 = st.columns(3)
            if s_col1.button("Performance Review", use_container_width=True): st.session_state.prompt_input = "Performance Review"
            if s_col2.button("Campaign Diagnosis", use_container_width=True): st.session_state.prompt_input = "Campaign Diagnosis"
            if s_col3.button("Trend Analysis", use_container_width=True): st.session_state.prompt_input = "Trend Analysis"
            
            prompt = st.text_area("Prompt", value=st.session_state.prompt_input, height=100)
            context = st.text_input("Additional Context (Optional)", placeholder="e.g., Target ROAS is 250%. Budget increased on July 10.")
            
            if st.button("Run Diagnosis", type="primary"):
                if not prompt:
                    st.warning("Please enter a prompt or select a suggestion.")
                else:
                    with st.spinner("Analyzing timeline data and generating insights..."):
                        time.sleep(1.5) # Fake processing time
                        
                        st.session_state.analysis_result = {
                            "title": prompt,
                            "date": "Jul 24",
                            "sources": "Google Ads, GA4",
                            "coverage": "2026-01-03 ~ 2026-07-24",
                            "context": context
                        }
                        st.session_state.archives.insert(0, st.session_state.analysis_result)
                        st.rerun()
                        
        # If result is generated, show Report Document
        else:
            res = st.session_state.analysis_result
            if st.button("← Back to Prompt"):
                st.session_state.analysis_result = None
                st.session_state.prompt_input = ""
                st.rerun()
                
            st.markdown(f"# {res['title']}")
            st.caption(f"Generated: {res['date']} | Sources: {res['sources']} | Coverage: {res['coverage']}")
            if res.get('context'):
                st.caption(f"Context: {res['context']}")
            st.divider()
            
            st.markdown("### 1. Executive Summary")
            st.write("Overall business performance maintained a stable trajectory over the analyzed timeline. However, efficiency metrics (ROAS and CPA) have shown slight degradation in the final two weeks of the coverage period (July 10 - July 24), despite a 15% increase in top-of-funnel traffic.")
            
            st.markdown("### 2. Key Findings")
            st.markdown("- **Traffic Volume (MoM):** Total sessions increased by 15%, primarily driven by Google Ads.\n- **Conversion Rate (Trend):** Dropped from an average of 2.4% in early Q2 to 1.9% in mid-July.\n- **Cost Efficiency:** CPA on Google Ads rose by 12% following the budget increase.")
            
            st.markdown("### 3. Root Causes")
            st.write("The recent decline in Conversion Rate correlates directly with the expanded audience targeting implemented around July 10. The appended dataset reveals that while these new segments generate cheaper clicks, they lack purchase intent, thereby diluting overall ROAS and increasing CPA.")
            
            st.markdown("### 4. Priority Actions")
            st.markdown("1. **Refine Targeting:** Immediately narrow the audience parameters on the newly expanded Google Ads campaigns to focus on higher-intent users.\n2. **Budget Reallocation:** Shift 15% of the current top-of-funnel budget back to proven remarketing channels.\n3. **Monitor Timeline:** Upload the next batch of data in 3 days to verify if the ROAS trend reverses toward the 250% target.")

# ==========================================
# 6. View: Archive
# ==========================================
def view_archive():
    nav_col1, nav_col2, nav_col3 = st.columns([8, 1, 1])
    with nav_col1:
        if st.button(f"← Projects ▾ {st.session_state.current_project}"):
            navigate("Projects")
    with nav_col2:
        if st.button("Workspace", use_container_width=True): navigate("Workspace")
    with nav_col3:
        st.button("Archive", use_container_width=True, disabled=True)
        
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
            st.markdown(f"# {arch.get('title', 'Analysis')}")
            st.caption(f"Generated: {arch.get('date', 'Unknown')} | Coverage: {arch.get('coverage', 'N/A')}")
            st.divider()
            
            st.markdown("### 1. Executive Summary")
            st.write("This is a historical record of the diagnosis run on " + arch.get('date', 'Unknown') + ". The core metrics indicated a stable environment but required immediate review on campaign spending.")
            
            st.markdown("### 2. Key Findings")
            st.write("- Historical traffic remained consistent.\n- Prior conversion rates were stable at 2.4%.")
            
            st.markdown("### 3. Root Causes")
            st.write("Identified minor seasonality impacts during the early weeks of the coverage timeline.")
            
            st.markdown("### 4. Priority Actions")
            st.write("1. Maintain current budget allocations.\n2. Upload fresh dataset next week to track MoM changes.")
            
            st.divider()
            if st.button("Re-run with latest timeline data"):
                st.session_state.prompt_input = arch.get('prompt', arch.get('title'))
                navigate("Workspace")
        else:
            st.info("Select a diagnosis from the history to view its details.")

# ==========================================
# 7. Router Logic
# ==========================================
if st.session_state.page == 'Projects':
    view_projects()
elif st.session_state.page == 'Workspace':
    view_workspace()
elif st.session_state.page == 'Archive':
    view_archive()
