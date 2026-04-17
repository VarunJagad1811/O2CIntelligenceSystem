import streamlit as st
import graphviz

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* The Main App Background - Deep Obsidian */
        .stApp { 
            background: linear-gradient(135deg, #09090b 0%, #111827 50%, #020617 100%); 
            color: #f8fafc; 
            font-family: 'Inter', sans-serif; 
        }

        /* Sidebar - Solid Dark Zinc */
        section[data-testid="stSidebar"] { 
            background-color: #020617 !important; 
            border-right: 1px solid rgba(255, 255, 255, 0.05); 
        }

        /* Headers - Sleeker Gradient */
        h1 { 
            background: linear-gradient(90deg, #f8fafc, #94a3b8); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-weight: 800 !important; 
            letter-spacing: -0.5px; 
            font-size: 2.8rem !important; 
        }
        
        h2, h3 { color: #f1f5f9 !important; font-weight: 700 !important; }

        /* Metric Containers - Glassmorphism on Black */
        .metric-container { 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            background: rgba(255, 255, 255, 0.02); 
            backdrop-filter: blur(20px); 
            border: 1px solid rgba(255, 255, 255, 0.08); 
            border-radius: 16px; 
            padding: 24px; 
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); 
            transition: all 0.3s ease; 
        }
        .metric-container:hover { 
            background: rgba(255, 255, 255, 0.05); 
            transform: translateY(-4px); 
            border-color: rgba(56, 189, 248, 0.4); 
        }
        
        .metric-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 600; margin-bottom: 8px; }
        .metric-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; }

        /* Tables and Dataframes */
        div[data-testid="stDataFrame"] { 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            border-radius: 12px; 
            background: #09090b; 
        }

        /* Report Box for AI Analysis */
        .report-box { 
            border: 1px solid rgba(56, 189, 248, 0.2); 
            background: rgba(255, 255, 255, 0.02); 
            border-radius: 12px; 
            padding: 25px; 
            margin-top: 15px; 
            font-size: 1.05rem; 
            line-height: 1.6;
        }

        /* Fix for Tabs selection color */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.03);
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_custom_metric(label, value, delta=None, delta_type="pos"):
    # Keeping the same logic but the CSS above will make it look "Black Obsidian"
    delta_html = f"<div class='metric-delta {'delta-pos' if delta_type == 'pos' else 'delta-neg'}'>{delta}</div>" if delta else ""
    st.markdown(f'<div class="metric-container"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{delta_html}</div>', unsafe_allow_html=True)

# --- LATENCY FIX: CACHING ---
# This tells Streamlit to memorize the graph. If 'is_risky' is True, it pulls the True graph from memory. 
# If False, it pulls the False graph. It completely skips the rendering engine math.
@st.cache_data(show_spinner=False)
def draw_process_graph(is_risky):
    dot = graphviz.Digraph()
    dot.attr(bgcolor='transparent', rankdir='LR', splines='true')
    dot.attr('node', shape='box', style='filled,rounded', fillcolor='#18181b', color='#3f3f46', fontcolor='#f8fafc', fontname="Inter")
    dot.attr('edge', fontname="Inter", color='#52525b', arrowsize='0.8')

    dot.node('A', '📥 Order Received', shape='oval', fillcolor='#09090b', color='#38bdf8', fontcolor='#38bdf8')
    dot.node('B', '🤖 AI Risk Check')
    dot.edge('A', 'B')

    if is_risky:
        dot.node('C', '⛔ MANUAL REVIEW', fillcolor='rgba(248, 113, 113, 0.1)', color='#f87171', fontcolor='#f87171')
        dot.node('D', 'Compliance Audit')
        dot.node('E', 'Final Decision')
        dot.edge('B', 'C', label=' Flagged', color='#f87171')
        dot.edge('C', 'D')
        dot.edge('D', 'E')
    else:
        dot.node('C', '✅ LOW RISK', fillcolor='rgba(74, 222, 128, 0.1)', color='#4ade80', fontcolor='#4ade80')
        dot.node('D', 'Warehouse Pick')
        dot.node('E', 'Shipped')
        dot.edge('B', 'C', label=' Cleared', color='#4ade80')
        dot.edge('C', 'D')
        dot.edge('D', 'E')
    return dot