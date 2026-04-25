import streamlit as st
import matplotlib.pyplot as plt

def plot_equity(history):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("Equity Curve")
    st.pyplot(fig)
