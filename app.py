import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Hardy Cross Table App", layout="wide")
st.title("💧 Hardy Cross Method – Table-Based Input")

st.markdown("""
This app allows you to simulate flow corrections in a pipe network using the Hardy Cross method with a simple editable table.
""")

# Sidebar: Set number of pipes
st.sidebar.header("🔢 Pipe Network Setup")
num_pipes = st.sidebar.number_input("Number of pipes", min_value=2, max_value=100, value=4)
iterations = st.sidebar.slider("Number of Hardy Cross iterations", 1, 50, 10)

# Default table
default_data = {
    "Pipe": [f"P{i+1}" for i in range(num_pipes)],
    "Start Node": [f"N{i}" for i in range(num_pipes)],
    "End Node": [f"N{i+1}" for i in range(num_pipes)],
    "Initial Q (m³/s)": [1.0] * num_pipes,
    "Resistance r": [10.0] * num_pipes,
    "Direction (1 or -1)": [1] * num_pipes
}

df_input = pd.DataFrame(default_data)
st.subheader("📋 Pipe Network Input Table")

edited_df = st.data_editor(df_input, use_container_width=True, num_rows="dynamic", key="input_table")

# Run simulation
if st.button("▶️ Run Hardy Cross"):
    try:
        Q = np.array(edited_df["Initial Q (m³/s)"], dtype=float)
        r = np.array(edited_df["Resistance r"], dtype=float)
        direction = np.array(edited_df["Direction (1 or -1)"], dtype=int)

        history = []
        for it in range(iterations):
            h_loss = direction * r * Q * np.abs(Q)
            sum_h = np.sum(h_loss)
            sum_dh = np.sum(2 * r * np.abs(Q))
            delta_Q = - sum_h / sum_dh
            Q += direction * delta_Q
            history.append((it + 1, delta_Q))

        # Create result DataFrame
        result_df = edited_df.copy()
        result_df["Final Q (m³/s)"] = Q

        st.success("✅ Hardy Cross calculation completed!")
        st.subheader("📊 Final Flows")
        st.dataframe(result_df)

        # Show delta Q evolution
        iter_df = pd.DataFrame(history, columns=["Iteration", "ΔQ (m³/s)"])
        st.subheader("📈 Iteration ΔQ Trend")
        st.line_chart(iter_df.set_index("Iteration"))

        # Excel export (optional - comment out if not using openpyxl)
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name="Results")
                iter_df.to_excel(writer, index=False, sheet_name="Iterations")
            st.subheader("📤 Download Excel Results")
            st.download_button("⬇️ Download Excel File", output.getvalue(), file_name="hardy_cross_results.xlsx")
        except ModuleNotFoundError:
            st.warning("openpyxl not found. Excel export skipped.")

        # CSV export
        st.subheader("📤 Download Results as CSV")
        csv_result = result_df.to_csv(index=False).encode('utf-8')
        csv_iterations = iter_df.to_csv(index=False).encode('utf-8')

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download Final Flows", data=csv_result, file_name="final_flows.csv", mime="text/csv")
        with col2:
            st.download_button("⬇️ Download Iteration Log", data=csv_iterations, file_name="iterations.csv", mime="text/csv")

        # Draw network
        st.subheader("🧩 Network Diagram")
        G = nx.DiGraph()
        for i, row in result_df.iterrows():
            G.add_edge(row["Start Node"], row["End Node"], label=f'{row["Final Q (m³/s)"]:.2f}')
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(8, 5))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
