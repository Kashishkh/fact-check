import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader
import os

# Initialize LLM (using GPT-4o-mini for cost-efficiency)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize search tool
search_tool = TavilySearchResults(max_results=3)  # Returns top 3 search results

# Prompt for extracting claims from PDF text
extract_prompt = ChatPromptTemplate.from_template("""
Extract all specific claims from this text. Claims include statistics, dates, financial figures, technical specs, or factual statements.
List them as a numbered list. For example:
1. Bitcoin is trading at $42,500.
2. GDP growth for 2025 is -1.5%.

Text: {text}
""")

# Prompt for verifying a single claim
verify_prompt = ChatPromptTemplate.from_template("""
Verify this claim using the search results.
Claim: {claim}
Search Results: {results}

Flag as:
- Verified: If it matches current data.
- Inaccurate: If it's close but outdated or slightly wrong (cite correct value).
- False: If no evidence or contradicted.

Explain briefly and cite sources.
""")

# Chain for extraction
extract_chain = extract_prompt | llm | StrOutputParser()

# Chain for verification (binds search tool to LLM)
verify_chain = verify_prompt | llm | StrOutputParser()

# Streamlit app
st.title("Fact-Checking Web App")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    st.write("PDF text extracted. Extracting claims...")
    
    # Extract claims
    claims_response = extract_chain.invoke({"text": text})
    claims = [claim.strip() for claim in claims_response.split("\n") if claim.strip() and claim[0].isdigit()]
    
    st.write(f"Found {len(claims)} claims:")
    for claim in claims:
        st.write(claim)
    
    results = []
    for claim in claims:
        st.write(f"Verifying: {claim}")
        
        # Search web for claim
        search_query = f"Current verification of: {claim}"
        search_results = search_tool.invoke({"query": search_query})
        results_str = "\n".join([res["content"] for res in search_results])
        
        # Verify with LLM
        verification = verify_chain.invoke({"claim": claim, "results": results_str})
        results.append(verification)
    
    st.write("Verification Report:")
    for i, result in enumerate(results):
        st.write(f"Claim {i+1}: {claims[i]}")
        st.write(result)
        st.write("---")
