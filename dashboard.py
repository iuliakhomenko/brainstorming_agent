import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import uuid
from datetime import datetime
import time
import random
from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 AI Brainstorming Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .idea-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .technique-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        cursor: pointer;
        transition: transform 0.2s;
    }

    .technique-card:hover {
        transform: translateY(-2px);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }

    .session-summary {
        background: #e8f4fd;
        border: 1px solid #b3d7ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'ideas' not in st.session_state:
    st.session_state.ideas = []
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'current_technique' not in st.session_state:
    st.session_state.current_technique = None
if 'problem_statement' not in st.session_state:
    st.session_state.problem_statement = ""


class MockBrainstormAgent:
    """Mock agent for demonstration - replace with real implementation"""

    def __init__(self):
        self.techniques = {
            "lateral_thinking": {
                "name": "🔄 Lateral Thinking",
                "description": "Challenge assumptions and think outside the box",
                "color": "#FF6B6B"
            },
            "scamper": {
                "name": "🛠️ SCAMPER",
                "description": "Systematic creative thinking technique",
                "color": "#4ECDC4"
            },
            "six_hats": {
                "name": "🎭 Six Thinking Hats",
                "description": "Explore ideas from multiple perspectives",
                "color": "#45B7D1"
            },
            "mind_mapping": {
                "name": "🗺️ Mind Mapping",
                "description": "Visual exploration of connections",
                "color": "#96CEB4"
            },
            "reverse_brainstorm": {
                "name": "🔄 Reverse Brainstorming",
                "description": "Think about how to cause the problem",
                "color": "#FFEAA7"
            }
        }

    def generate_ideas(self, problem: str, technique: str, context: Dict) -> List[Dict]:
        """Generate mock ideas based on technique"""
        base_ideas = {
            "lateral_thinking": [
                "What if we eliminated the problem entirely?",
                "How would a child approach this?",
                "What's the opposite solution?",
                "How would aliens solve this?",
                "What if money wasn't a factor?"
            ],
            "scamper": [
                "Substitute: Replace key components",
                "Combine: Merge with another solution",
                "Adapt: Borrow from other industries",
                "Modify: Make it bigger/smaller",
                "Put to other uses: Different applications"
            ],
            "six_hats": [
                "White Hat: Gather more data",
                "Red Hat: Trust your gut feeling",
                "Black Hat: Identify potential risks",
                "Yellow Hat: Focus on benefits",
                "Green Hat: Generate alternatives"
            ],
            "mind_mapping": [
                "Central theme connection",
                "Branch out to related concepts",
                "Find unexpected links",
                "Cluster similar ideas",
                "Identify missing connections"
            ],
            "reverse_brainstorm": [
                "How to make the problem worse",
                "Opposite of desired outcome",
                "Eliminate successful elements",
                "Add more complexity",
                "Ignore user needs completely"
            ]
        }

        ideas = []
        for i, base_idea in enumerate(base_ideas.get(technique, [])):
            ideas.append({
                "id": str(uuid.uuid4()),
                "title": f"{base_idea} - {problem}",
                "description": f"Detailed exploration of {base_idea.lower()} applied to: {problem}",
                "technique": technique,
                "creativity_score": random.randint(6, 10),
                "feasibility_score": random.randint(5, 9),
                "impact_score": random.randint(6, 10),
                "total_score": random.randint(17, 28),
                "timestamp": datetime.now(),
                "tags": random.sample(["innovative", "practical", "scalable", "cost-effective", "quick-win"], 2)
            })

        return ideas


# Initialize mock agent
if 'agent' not in st.session_state:
    st.session_state.agent = MockBrainstormAgent()


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AI Brainstorming Agent</h1>
        <p>Unleash your creativity with AI-powered brainstorming techniques</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for session management and technique selection
    with st.sidebar:
        st.header("🎯 Session Control")

        # Problem input
        problem = st.text_area(
            "What challenge are you trying to solve?",
            value=st.session_state.problem_statement,
            placeholder="e.g., How can we increase customer retention?",
            height=100
        )

        if problem != st.session_state.problem_statement:
            st.session_state.problem_statement = problem

        # Context inputs
        st.subheader("📋 Context")
        col1, col2 = st.columns(2)

        with col1:
            industry = st.selectbox(
                "Industry",
                ["Technology", "Healthcare", "Finance", "Retail", "Education", "Manufacturing", "Other"]
            )

            timeline = st.selectbox(
                "Timeline",
                ["Urgent (1 week)", "Short-term (1 month)", "Medium-term (3 months)", "Long-term (6+ months)"]
            )

        with col2:
            budget = st.selectbox(
                "Budget",
                ["Limited ($0-1K)", "Moderate ($1K-10K)", "Substantial ($10K+)", "No constraints"]
            )

            team_size = st.selectbox(
                "Team Size",
                ["Individual", "Small team (2-5)", "Large team (6+)", "Department", "Company-wide"]
            )

        # Technique selection
        st.subheader("🛠️ Brainstorming Techniques")

        techniques = st.session_state.agent.techniques
        selected_technique = st.radio(
            "Choose a technique:",
            list(techniques.keys()),
            format_func=lambda x: techniques[x]["name"]
        )

        # Generate ideas button
        if st.button("🚀 Generate Ideas", type="primary", use_container_width=True):
            if problem:
                with st.spinner("🧠 AI is brainstorming..."):
                    time.sleep(2)  # Simulate processing time
                    context = {
                        "industry": industry,
                        "timeline": timeline,
                        "budget": budget,
                        "team_size": team_size
                    }

                    new_ideas = st.session_state.agent.generate_ideas(
                        problem, selected_technique, context
                    )

                    st.session_state.ideas.extend(new_ideas)
                    st.session_state.current_technique = selected_technique

                    # Add to session history
                    st.session_state.session_history.append({
                        "technique": selected_technique,
                        "ideas_generated": len(new_ideas),
                        "timestamp": datetime.now()
                    })

                st.success(f"✨ Generated {len(new_ideas)} new ideas!")
                st.rerun()
            else:
                st.error("Please describe your challenge first!")

        # Session stats
        if st.session_state.ideas:
            st.subheader("📊 Session Stats")
            total_ideas = len(st.session_state.ideas)
            avg_score = sum(idea["total_score"] for idea in st.session_state.ideas) / total_ideas
            techniques_used = len(set(idea["technique"] for idea in st.session_state.ideas))

            col1, col2, col3 = st.columns(3)
            col1.metric("Ideas", total_ideas)
            col2.metric("Avg Score", f"{avg_score:.1f}")
            col3.metric("Techniques", techniques_used)

        # Clear session
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in ['ideas', 'session_history', 'current_technique', 'problem_statement']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Main content area
    if not st.session_state.ideas:
        show_welcome_screen()
    else:
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💡 Ideas Dashboard",
            "📊 Analytics",
            "🌐 Mind Map",
            "☁️ Word Cloud",
            "📈 Session Timeline"
        ])

        with tab1:
            show_ideas_dashboard()

        with tab2:
            show_analytics()

        with tab3:
            show_mind_map()

        with tab4:
            show_word_cloud()

        with tab5:
            show_session_timeline()


def show_welcome_screen():
    """Display welcome screen with technique explanations"""
    st.markdown("## 🌟 Welcome to AI Brainstorming!")

    st.markdown("""
    Get started by:
    1. **Describing your challenge** in the sidebar
    2. **Setting the context** (industry, timeline, budget)
    3. **Choosing a brainstorming technique**
    4. **Generating ideas** with AI assistance
    """)

    # Technique showcase
    st.subheader("🛠️ Available Techniques")

    techniques = st.session_state.agent.techniques
    cols = st.columns(3)

    for i, (key, technique) in enumerate(techniques.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="technique-card" style="background: {technique['color']};">
                <h4>{technique['name']}</h4>
                <p>{technique['description']}</p>
            </div>
            """, unsafe_allow_html=True)


def show_ideas_dashboard():
    """Main ideas dashboard with filtering and sorting"""
    st.header("💡 Generated Ideas")

    if not st.session_state.ideas:
        st.info("No ideas generated yet. Use the sidebar to get started!")
        return

    # Filtering and sorting controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Total Score", "Creativity", "Feasibility", "Impact", "Recent"]
        )

    with col2:
        filter_technique = st.selectbox(
            "Filter by technique:",
            ["All"] + list(set(idea["technique"] for idea in st.session_state.ideas))
        )

    with col3:
        min_score = st.slider("Minimum score:", 0, 30, 0)

    with col4:
        show_count = st.selectbox("Show:", [10, 25, 50, "All"])

    # Filter and sort ideas
    filtered_ideas = st.session_state.ideas.copy()

    if filter_technique != "All":
        filtered_ideas = [idea for idea in filtered_ideas if idea["technique"] == filter_technique]

    filtered_ideas = [idea for idea in filtered_ideas if idea["total_score"] >= min_score]

    # Sort ideas
    sort_key_map = {
        "Total Score": "total_score",
        "Creativity": "creativity_score",
        "Feasibility": "feasibility_score",
        "Impact": "impact_score",
        "Recent": "timestamp"
    }

    filtered_ideas.sort(
        key=lambda x: x[sort_key_map[sort_by]],
        reverse=True
    )

    if show_count != "All":
        filtered_ideas = filtered_ideas[:show_count]

    # Display ideas
    for i, idea in enumerate(filtered_ideas):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"""
                <div class="idea-card">
                    <h4>💡 {idea['title']}</h4>
                    <p>{idea['description']}</p>
                    <div style="margin-top: 10px;">
                        <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 5px;">
                            {st.session_state.agent.techniques[idea['technique']]['name']}
                        </span>
                        {' '.join([f'<span style="background: #f3e5f5; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 5px;">{tag}</span>' for tag in idea['tags']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Score visualization
                fig = go.Figure(go.Scatterpolar(
                    r=[idea["creativity_score"], idea["feasibility_score"], idea["impact_score"]],
                    theta=["Creativity", "Feasibility", "Impact"],
                    fill='toself',
                    name=f"Idea {i + 1}"
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10])
                    ),
                    showlegend=False,
                    height=200,
                    margin=dict(l=0, r=0, t=20, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⭐ Favorite", key=f"fav_{idea['id']}"):
                        st.success("Added to favorites!")

                with col_btn2:
                    if st.button("📝 Develop", key=f"dev_{idea['id']}"):
                        show_idea_development(idea)


def show_analytics():
    """Analytics dashboard with charts and insights"""
    st.header("📊 Brainstorming Analytics")

    if not st.session_state.ideas:
        st.info("Generate some ideas first to see analytics!")
        return

    # Create analytics dataframe
    df = pd.DataFrame(st.session_state.ideas)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Ideas",
            len(df),
            delta=len(df) - len(
                st.session_state.session_history[0]["ideas_generated"]) if st.session_state.session_history else 0
        )

    with col2:
        avg_creativity = df["creativity_score"].mean()
        st.metric("Avg Creativity", f"{avg_creativity:.1f}/10")

    with col3:
        avg_feasibility = df["feasibility_score"].mean()
        st.metric("Avg Feasibility", f"{avg_feasibility:.1f}/10")

    with col4:
        top_score = df["total_score"].max()
        st.metric("Top Score", f"{top_score}/30")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Score distribution
        fig = px.histogram(
            df,
            x="total_score",
            nbins=10,
            title="Score Distribution",
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Technique comparison
        technique_stats = df.groupby("technique").agg({
            "creativity_score": "mean",
            "feasibility_score": "mean",
            "impact_score": "mean"
        }).round(1)

        fig = px.bar(
            technique_stats.reset_index(),
            x="technique",
            y=["creativity_score", "feasibility_score", "impact_score"],
            title="Technique Performance",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Score Correlations")
    correlation_cols = ["creativity_score", "feasibility_score", "impact_score", "total_score"]
    corr_matrix = df[correlation_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Score Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top ideas table
    st.subheader("🏆 Top Ideas")
    top_ideas = df.nlargest(5, "total_score")[
        ["title", "technique", "creativity_score", "feasibility_score", "impact_score", "total_score"]]
    st.dataframe(top_ideas, use_container_width=True)


def show_mind_map():
    """Interactive mind map visualization"""
    st.header("🌐 Ideas Mind Map")

    if not st.session_state.ideas:
        st.info("Generate ideas to see the mind map!")
        return

    # Create network graph
    G = nx.Graph()

    # Add central problem node
    problem = st.session_state.problem_statement[:50] + "..." if len(
        st.session_state.problem_statement) > 50 else st.session_state.problem_statement
    G.add_node("PROBLEM", label=problem, size=30, color="#FF6B6B")

    # Add technique nodes
    techniques_used = list(set(idea["technique"] for idea in st.session_state.ideas))
    for technique in techniques_used:
        technique_name = st.session_state.agent.techniques[technique]["name"]
        G.add_node(technique, label=technique_name, size=20, color="#4ECDC4")
        G.add_edge("PROBLEM", technique)

    # Add idea nodes
    for i, idea in enumerate(st.session_state.ideas[:15]):  # Limit for readability
        idea_label = idea["title"][:30] + "..." if len(idea["title"]) > 30 else idea["title"]
        node_size = 10 + (idea["total_score"] / 30) * 15  # Size based on score
        G.add_node(f"idea_{i}", label=idea_label, size=node_size, color="#96CEB4")
        G.add_edge(idea["technique"], f"idea_{i}")

    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Create plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node].get('label', node))
        node_color.append(G.nodes[node].get('color', '#97C2FC'))
        node_size.append(G.nodes[node].get('size', 10))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Ideas Mind Map",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Nodes sized by idea score",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="#888", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    st.plotly_chart(fig, use_container_width=True)


def show_word_cloud():
    """Generate and display word cloud from ideas"""
    st.header("☁️ Ideas Word Cloud")

    if not st.session_state.ideas:
        st.info("Generate ideas to see the word cloud!")
        return

    # Combine all idea text
    all_text = " ".join([
        idea["title"] + " " + idea["description"]
        for idea in st.session_state.ideas
    ])

    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_text)

        # Convert to image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        # Save to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)

        st.image(img_buffer, use_column_width=True)

        # Word frequency analysis
        st.subheader("🔤 Most Common Words")
        word_freq = wordcloud.words_

        if word_freq:
            freq_df = pd.DataFrame(
                list(word_freq.items())[:10],
                columns=["Word", "Frequency"]
            )

            fig = px.bar(
                freq_df,
                x="Frequency",
                y="Word",
                orientation='h',
                title="Top 10 Words in Ideas"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate word cloud: {e}")
        st.info("Install wordcloud package: pip install wordcloud")


def show_session_timeline():
    """Show session timeline and progress"""
    st.header("📈 Session Timeline")

    if not st.session_state.session_history:
        st.info("No session history available yet!")
        return

    # Timeline data
    timeline_data = []
    cumulative_ideas = 0

    for i, session in enumerate(st.session_state.session_history):
        cumulative_ideas += session["ideas_generated"]
        timeline_data.append({
            "Step": i + 1,
            "Technique": st.session_state.agent.techniques[session["technique"]]["name"],
            "Ideas Generated": session["ideas_generated"],
            "Cumulative Ideas": cumulative_ideas,
            "Time": session["timestamp"].strftime("%H:%M:%S")
        })

    timeline_df = pd.DataFrame(timeline_data)

    # Timeline chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=timeline_df["Step"],
            y=timeline_df["Ideas Generated"],
            name="Ideas per Technique",
            marker_color="#667eea"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Step"],
            y=timeline_df["Cumulative Ideas"],
            mode="lines+markers",
            name="Cumulative Ideas",
            line=dict(color="#FF6B6B", width=3)
        ),
        secondary_y=True,
    )

    fig.update_layout(title="Session Progress Timeline")
    fig.update_xaxes(title_text="Session Step")
    fig.update_yaxes(title_text="Ideas Generated", secondary_y=False)
    fig.update_yaxes(title_text="Total Ideas", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Session details table
    st.subheader("Session Details")
    st.dataframe(timeline_df, use_container_width=True)

    # Session insights
    st.subheader("💡 Session Insights")

    if len(timeline_df) > 1:
        most_productive = timeline_df.loc[timeline_df["Ideas Generated"].idxmax()]
        avg_ideas_per_technique = timeline_df["Ideas Generated"].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Most Productive Technique:**  
            {most_productive['Technique']} - {most_productive['Ideas Generated']} ideas
            """)

        with col2:
            st.info(f"""
            **Average Ideas per Technique:**  
            {avg_ideas_per_technique:.1f} ideas
            """)


def show_idea_development(idea):
    """Show detailed idea development interface"""
    st.subheader(f"🚀 Developing: {idea['title']}")

    with st.expander("Idea Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Description:**")
            st.write(idea['description'])

            st.write("**Tags:**")
            st.write(", ".join(idea['tags']))

        with col2:
            # Implementation steps
            st.write("**Next Steps:**")
            steps = [
                "Research feasibility",
                "Create prototype",
                "Gather feedback",
                "Refine concept",
                "Plan implementation"
            ]

            for i, step in enumerate(steps):
                st.checkbox(f"{i + 1}. {step}", key=f"step_{idea['id']}_{i}")

    # Notes section
    notes = st.text_area(
        "Development Notes:",
        placeholder="Add notes about this idea...",
        key=f"notes_{idea['id']}"
    )

    if st.button("💾 Save Development Plan", key=f"save_{idea['id']}"):
        st.success("Development plan saved!")


if __name__ == "__main__":
    main()