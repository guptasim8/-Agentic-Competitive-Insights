# 👨🏻‍💻 Use case: Competitive Analysis
## Table of Contents
- [Architecture](#-architecture)
- [Use Case Description](#use-case-description)
- [Pre-requisites](#pre-requisites)
- [Lab Steps Overview](#lab-steps-overview)
- [Lab Instructions](#lab-instructions)
  - [Connect to your assigned Watsonx Orchestrate instance](#connect-to-your-assigned-watsonx-orchestrate-instance)
  - [Create Identifier Agent](#create-identifier-agent)
  - [Create Comp Analysis Agent](#create-comp-analysis-agent)
  - [Create ABC Robots Agent](#create-abc-robots-agent)
  - [Create Master Agent](#create-master-agent)
  - [Experience Agents in Action](#experience-agents-in-action)

## 🏛 Architecture  
![Competitive Analysis Architecture drawio](https://github.ibm.com/user-attachments/assets/39d11294-f75d-482d-a94d-80e0511a5aa7)

## Use Case Description

ABC Robots plans to implement an AI-powered Competitive Intelligence System to automate market research and competitor analysis. This system will help sales teams quickly identify and position their products against competitors, overcoming the inefficiencies of manual research and outdated insights. The goal is to create an AI-enabled system that supports competitive analysis and market research by:

* Extracting products from the company’s product catalog
* Identifying and extracting key features of each product
* Searching for competitor products based on key attributes
* Generating a structured competitive comparison table with price, features, and differentiators

By automating these tasks, the company aims to accelerate sales processes, improve data accuracy, and enable sales teams to make informed decisions faster.

## Pre-requisites

**Instructors**: 
- Check the corresponding [Instructor's guide](https://github.ibm.com/skol/agentic-ai-client-bootcamp-instructors/tree/main/usecase-setup/competitive-analysis) to set up all environments and backend services.
- Provide the URL for the deployed MCP Server

**Participants**:
- Validate that you have access to the right TechZone environment for this lab.
- Complete the [environment-setup](../../environment-setup) guide for steps on API key creation and project setup.
- Ensure the instructor has provided the URL to connect to the MCP server
- Ensure your instructor has provided the `Vaccum_cleaners_v2.docx` file to be loaded as knowledge
- Ensure your instructor has provided the `abc-robots-website-final.zip` file to embed your chat into a website.
- Ensure that your instructor has provided all necessary credentials.

## Lab Steps Overview
1. Connect to watsonx Orchestrate
2. Create the Identifier Agent - uses MCP Tool extract_from_image
3. Create the Comparison Analysis Agent - uses MCP Tool search_and_review_high_rated_products
4. Create the ABC Robots Agent - RAG agent
5. Create the Master Agent - orchestrator agent

Let's get started. 
## Lab Instructions
### Connect to your assigned Watsonx Orchestrate instance

1. Log in to IBM Cloud (cloud.ibm.com). Navigate to the top-left hamburger menu, then to **Resource List**. Open the **AI/Machine Learning** section. You should see a **watsonx Orchestrate** service. Click to open it.

   ![Watsonx Orchestrate service](./assets/i1.png)

2. Click the **Launch watsonx Orchestrate** button:

   ![Launch Watsonx Orchestrate](./assets/i2.png)
  
### Create Identifier Agent

The Identifier Agent recognizes a product from an image. It relies on a service predeployed in an MCP server that invokes a watsonx.ai hosted multi-modal model for image recognition. Let's integrate this service into a watsonx Orchestrate agent.

> [!TIP]
> An MCP server is a live deployment of an implementation of the Model Context Protocol (MCP), which gives standardized access to external tools, data, and services securely and consistently.

1. Go to the watsonx Orchestrate home page, click on the hamburger menu (☰), select **Build**, and then **Agent Builder**.
![Agent Builder](assets/BAP_1.png)

2. Click on the **Create agent** button.
![Create Agent](assets/BAP_2.png)

3. Select **Create from scratch**, add the following information:
   
   **Name**:
   ```
   Identifier Agent
   ```
   **Description**:
        
   ```
   This agent will extract the brand name and product name from the image.
   ```

   Click on **Create** button

   ![Create from scratch](assets/id_create.png)

4. Go to the **Toolset** section and click on **Add tool**:
   ![Add tool](assets/id_tool.png)

5. Select **Add from file or MCP server**:
   ![Add MCP Server](assets/id_add_mcp.png)

6. Select **Import from MCP server**:
   ![Import from MCP Server](assets/id_import_mcp.png)

7. On the **Import or remove tools from MCP server** window, click **Add MCP server**:
   ![Add MCP Server](assets/id_add_mcp_button.png)

8. On the **Add MCP server** window add the following parameters and then click **Connect**:

   - Server Name:
     ```
     mcp-competitive-tools
     ```
   - Get the MCP Server URL from your instructor and fill it in the Install command as follows:
     ```
     uvx mcp-proxy [MCP-SERVER-URL]/sse
     ```
     > e.g., `uvx mcp-proxy https://remote-mcp-tools.20sp7brq6u6i.us-south.codeengine.appdomain.cloud/sse`

   ![Server Details](assets/id_server_details.png)

9. You should see a `Connection successful` message underneath. Click **Done**:
   ![Connection Successful](assets/id_connect.png)

10. On the **Import or remove tools from MCP server** window, toggle on the `extract_from_image` tool to **On** then close the window.
   ![Toogle tool](assets/id_image_toggle.png)

11. Now let's define what the agent is supposed to do in the **Behavior** section. Let's be specific in terms of the format we want to get from the image recognition agent. Try the following **Instructions**:

    ```
    I will share an image URL. Please use the extract_from_image tool to parse the content. From the extracted text or context, identify the exact product name and brand name from the reference image. This information will be used by the Competitive Analysis Agent.

    Return the result in this format:

    Brand Name :
   
    Product Name:
    ```
    ![Behavior](assets/id_behavior.png)

12. Let's test the agent on the spot by loading the following image. We will pass the corresponding URL for the agent to recognize the product:
   <img src="https://m.media-amazon.com/images/I/613mvDKX1hL._AC_SL1500_.jpg" width="400">
  
13. Type the following query in the chat on the right side in the **Preview** window:

    ```
    Tell me what product is in this image https://m.media-amazon.com/images/I/613mvDKX1hL._AC_SL1500_.jpg
    ```
   
   ![Behavior](assets/id_test.png)

14. You can now deploy the agent. Click the **Deploy** button and then the **Deploy** again on the **Pre-deployment summary** window:

    ![Deploy](assets/id_deploy.png)
    ![Deploy](assets/id_deploy_2.png)

    You will now see that your agent is **Live** and you can chat with it directly.  We will demonstrate this later in the lab.

    ![Deploy](assets/id_live.png)

## Create Comp Analysis Agent

Now, let's create the Comp Analysis Agent. This one allows to get information from products in the market using Google Search and Google Shopping APIs. 

We use the SerpAPI to invoke these services and then exposed them as tools in the MCP server for easier invocation from the Agent in watsonx Orchestrate.

1. Click on the **Manage Agents** link in the breadcrumb menu at top left.
   ![Manage Agent](assets/id_manage_agent.png)

2. Select the **Create agent** button

   ![Create Agent](assets/comp_create.png)

3. Select **Create from scratch**.
   
   **Name**:
   ```
   Comp Analysis Agent
   ```
   **Description**:
        
   ```
   Provides elaborate, very detailed analysis using a tool.
   ```

   Click on **Create** button.
   <!--
   ![Create from scratch](assets/comp_create_agent.png)
   -->
4. Set the model to `llama-3-405b-instruct`. This model is more appropriate to handle queries than the vision one.

   ![Create from scratch](assets/comp_model.png)

5. Go to the **Toolset** section and click on **Add tool** button.
   ![Add tool](assets/comp_tool.png)

6. Select **Add from file or MCP server**.
   ![Add MCP Server](assets/comp_add_mcp.png)

7. Select **Import from MCP server**.
   ![Import MCP Server](assets/comp_import_mcp.png)

8. Since we connected up the MCP server in the previous Agent steps, you will now have a choice to choose your MCP server. On the **Import or remove tools from MCP server** window, choose the **Select MCP server** dropdown and select the `mcp-competitive-tools` server.
   ![Select MCP Server](assets/comp_select_mcp.png)

9. Once the available tools are displayed, toggle the `search_and_review_high_rated_products` tool to **On** then close the window.
   ![Toogle tool](assets/comp_search_toggle.png)

10. In the **Behavior** section, add the following to the **Instructions** text field:

    ```
    When given a product name, use the search_and_review_high_rated_products tool to retrieve the content for the user's query.
    ```
    ![Behavior](assets/comp_behavior.png)

11. Now you can test your agent. Add the following query to the text input box on bottom right of **Preview** window:

    ```
    Give me details of the Dreame L10 Pro.
    ```
   
<!--
   ![Test](assets/comp_test.png)

1. You can also see the reasoning the agent used to achieve the response. Click the **Deploy** button at the top right to deploy the agent to the live environment:

   ![Steps](assets/comp_steps.png)
-->   
12. Click the **Deploy** button at the top right to deploy the agent to the live environment. Click the **Deploy** button again on the **Pre-deployment summary** window.

    ![Deploy](assets/comp_deploy_2.png)

    You will now see that your agent is **Live** and you can chat with it directly. 

    ![Live](assets/comp_live.png)

## Create ABC Robots Agent

This agent will pull information from ABC Robots catalog of product. In real life, product information might be in a database or other type of enterprise repository. For simplicity, we will upload a PDF with the product catalog. The following products are in the catalog: 

<br>
<br>

<img width="1187" alt="Screenshot 2025-10-01 at 10 58 12 AM" src="https://github.com/guptasim8/-Agentic-Competitive-Insights/blob/main/assets/37dbc688-a5b7-4835-8564-07637ea5a57b.png">

<br>
<br>

1. Click on the **Manage Agents** link in the breadcrumb menu at top left.
![Manage Agent](assets/comp_manage.png)

2. Select the **Create agent** button

   ![Create Agent](assets/abc_create_agent.png)

3. Select **Create from scratch**.
   
   **Name**:
   ```
   ABC Robots Agent
   ```
   **Description**:
        
   ```
   This agent will answer questions using the uploaded knowledge base. Always interpret the user’s input as a query to the knowledge base. The user will ask queries related to robotic vacuum cleaners only as the knowledge base contains that information.
   ```

   Click on **Create** button.
<!--
   ![Create from scratch](assets/abc_create.png)
-->
4. In the **Knowledge Source** section and click on the **Choose knowledge** button.
   ![Knowledge](assets/abc_knowledge.png)

5. After clicking the **Choose knowledge** button, a pop-up window will appear. Select **Upload files**, then click **Next**.
   ![Knowledge](assets/abc_upload.png)

6. Upload the provided [Vacuum_cleaners_v2.docx](assets/Vaccum_cleaners_v2.docx) document. And click the **Next** button
   ![Knowledge](assets/abc_file_upload.png)

7. Add the description below in the **Description** field, and then click **Save**.
   **Description:**
   ```
   This knowledge document contains all the product-related information for ABC Robots. All queries related to the product will be addressed using this document as the primary source.
   ```
   ![Knowledge](assets/abc_knowledge_source.png)

8. After completing all the above steps, your knowledge source will be added and will appear as shown in the image below.
   ![Upload file](assets/abc_knowledge_results.png)

9. Click on the **Edit Knowledge Settings** button and change the **Response** to **Dynamic (Preview)** and the **Maximum Search Results** to 10, then click the **Save** button.
   ![settings](assets/abc_knowledge_settings.png)
   
10. In the **Behavior** section, add the following to the **Instructions** text field:

    ```
    Specs query → Structured summary of that product from KB (exact text, no paraphrasing).
    Comparison query → Generate a side-by-side comparison from KB. Each distinct function, feature, or specification must be presented as a separate row in the table (e.g., individual rows for "Cleaning Modes", "Battery Life", "Navigation System", "Dustbin Capacity", etc.). Do not consolidate multiple features into grouped rows like "Core Functions" or "Main Features".
    Competitive analysis vs KB → Compare a given product (if in KB) vs all KB products.
    No relevant data → Output strictly:
    
    The information required cannot be found in the current knowledge base. Please upload the relevant data in a supported format (e.g., CSV, TSV, or text document).

    ```
    ![Upload file](assets/abc_behavior.png)

11. Now you can test your agent. Ask questions about ABC robots products, or specific information for any product.

    ```
    Give me the list of products for ABC robots
    ```

    ```
    Give me information for the Nimbus S7
    ```

12. Click the **Deploy** button at the top right to deploy the agent to the live environment. Click the **Deploy** button again on the **Pre-deployment summary** window:

    ![Deploy](assets/abc_deploy_2.png)

    You will now see that your agent is **Live** and you can chat with it directly. 

## Create Master Agent 

1. Click on the **Manage Agents** link in the breadcrumb menu at top left.
<!--![Manage Agent](assets/abc_manage.png)-->

2. Select the **Create agent** button
<!--   ![Create Agent](assets/master_create.png) -->

3. Select **Create from scratch**.
   
   **Name**:
   ```
   Master Agent
   ```
   **Description**:
        
   ```
   You are an intelligent assistant; you have the capability of choosing agents based on the user's request. Ensure to follow the behavior strictly.
   ```

   Click on **Create** button.

   ![Create from scratch](assets/master_create_agent.png)

4. Set the model to `llama-3-405b-instruct`.
   ![Model](assets/master_model.png)

5. In the **Agents** section, click on the **Add agent** button.

    ![Add Agent](./assets/master_add_agent.png)

6. Click **Add from local instance**.
   ![Add Agent](./assets/master_add_local.png)

7. Select the **ABC Robots Agent**, the **Comp Analysis Agent** and the **Identifier Agent**, then click the **Add to Agent** button.

   ![Add Agents](./assets/master_add_agents.png)

8. In the **Behavior** section, add the following to the **Instructions** text field:

    ```
    The Master Agent does not answer queries directly. Its only responsibility is to route queries to the appropriate specialised agents, manage state between interactions, and ensure downstream queries are context-aware.
    Routing Rules
    Image-based Product Identification:
    If a user query contains an image or explicitly asks “Tell me what product is in the image,” route the query to the Identifier Agent.
    After the response is received from the Identifier Agent, display the brand name and model name and ask the user - “Would you like me to pull information for this product?”.
    If the user answers "Yes," provide the output from the Identifier Agent to the comp-analysis-ag and, from the response received, analyze and understand the response to provide a detailed and very long answer, divided into specifications, features, and reviews only. Ensure there is no none response in the headers.
    Customer Perception Summary:
    If a user queries "Give me the summary of how this product is perceived by customers, broken down into good and bad," then provide a detailed breakdown of reviews divided into good and bad from the response already received from comp-analysis-ag.
    Competitive Analysis against Knowledge Base:
    If a user queries “Do a competitive analysis of this product against the knowledge base,” then a detailed comparison needs to be done between the response already received from comp-analysis-ag and the knowledge base from ABC Robots (all the products). Extract all information from the knowledge base and compare all features, price, and ratings against comp-analysis-ag. Understand and analyze, then provide that response in a detailed and long tabular manner. Decide on columns nicely for its table, making sure all comparisons are under one table.
    Product-specific Queries (ABC Robots):
    If a user queries “Give me a comparison between Aerowash X1 and Nimbus S7,” or “Give me the specifications of Aerowash X1,” "Give me the products of ABC robots" (in products of ABC, just provide a list of products and the table should be nicely proportionate), or similar queries like this with other product names too, then redirect it to ABC Robots.
    Display features, pricing, or any other relevant detailed answers along with their respective sub-headings in a detailed and a long tabular manner vertically.
    The answer should start with "Based on the knowledge base, here is the comparison between HydraClean V9 and Nimbus S7" (or relevant products), in a tabular format vertically only for other products too. Analyze the products and then provide columns and their rows of comparison nicely.
    At the end, give your detailed and a long comparison analysis too after the tabular comparison.
    Competitive Analysis with Top Competitors:
    If a user queries “Give me a competitive analysis of HydraClean v9 and the top competitors in the market” (there can be any product instead of HydraClean v9), route the query to comp-analysis-ag and pass this query “HydraClean v9 and top competitors in the market” only.
    Once a response is received, understand and analyze it, and provide a new tabular response (decide on columns and rows smartly). In the end, give your detailed analysis.
   ```

9. Click the **Deploy** button at the top right to deploy the agent to the live environment:

   ![Steps](assets/master_deploy.png)
   
10. Click the **Deploy** button again on the **Pre-deployment summary** window:

    ![Deploy](assets/master_deploy_agent.png)

    You will now see that your agent is **Live** and you can chat with it directly. 
   
## Experience Agents in Action
Follow the steps above, then try interacting with the use case using these sample queries:

1. Go to the hamburger menu and select **Chat**
   ![chat](assets/chat.png)
  
2. Select the **Master Agent** from the dropdown menu, and you should be good to go.
   ![Master Agent](assets/chat_master.png)

3. Ask the following queries which should be routed to the ABC Robots Agent:
   ```
   Show me the list of products by ABC robots
   ```
   ![ABC Agent Response](assets/chat_q1.png)  
   ```
   Give me the specifications of Aerowash X1
   ```
   ![ABC Agent Response](assets/chat_q2.png)
   ```
   Give me a comparison table between Aerowash X1 and HydraClean v9 broken down into individual features
   ```

   <img width="684" alt="Screenshot 2025-10-04 at 11 21 08 AM" src="https://github.ibm.com/skol/agentic-ai-client-bootcamp/assets/12043/1555eefe-73fb-4812-94b7-452bc8e22123">

## Now Click on the blue icon to create a New Chat and start a new conversation:
![New Chat Image](assets/image41.png)


4. To identify product information from an image, ask:
   ```
   Tell me what product is in this image https://m.media-amazon.com/images/I/613mvDKX1hL._AC_SL1500_.jpg
   ```
   
   You will then be asked "Would you like me to pull information for this model?". Respond with `yes`:
   
   ![Comparison Agent Response](assets/chat_q5.png)
   
5. To do some competitive analysis, try these prompts:
   ```
   Give me the specifications of the product in the image
   ```
   <!--  ![ABC Agent Response](assets/chat_q6.png) -->

   ```
   Give me a summary of how this product is perceived by customers
   ```
   ![ABC Agent Response](assets/chat_q7.png)

   ```
   Give me a competitive analysis of the product in the image against each of the products in the catalog for ABC robots. Break it down by individual features in a table. Include Laundry and Dishwashing as well. Put it all together into a single table.
   ```

   <img width="685" alt="Screenshot 2025-10-04 at 11 26 16 AM" src="https://github.ibm.com/skol/agentic-ai-client-bootcamp/assets/12043/cdde7a14-cd88-4f2a-bdcd-7c8869d9ee53">

   <!-- 
   ```
   Give me the top vacuum robots in the market
   ```
   ![ABC Agent Response](assets/chat_q9.png)

   ```
   Give me a competitive analysis of the ABC HydraClean v9 vs the products above
   ```
   ![ABC Agent Response](assets/chat_q10.png)

   -->

3.2.6	Deployment
Now you will deploy this chat into ABC robots’ internal website so ABC robots employees are able to perform competitive analysis. 

1.	Go to **Manage Agents** and then to your Master Agent.
2.	Scroll down to **Channels** and click on **Embedded Agent**
<img width="700" alt="embedding" src="https://github.ibm.com/skol/agentic-ai-client-bootcamp/assets/12043/0a57a2ec-53ee-4266-bb4d-4fefd5bafd48">

3.	Copy the code by clicking on the **Copy to Clipboard** button on the top right of the code block starting with <script>.
4.	Extract the ABC Robots website ZIP file provided by your instructor.
5.	Edit the **index.html** file using your favorite text editor or development tool.

   <img width="700" alt="deployment-code" src="https://github.ibm.com/skol/agentic-ai-client-bootcamp/assets/12043/f2d323bc-f693-4d83-a8f2-71bd485de941">

6.	Scroll all the way down and paste it right before the </body> tag.
 
•	Save your changes and open the **index.html** file with your browser. You will see something like this:
<img width="700" alt="robots-website" src="https://github.ibm.com/skol/agentic-ai-client-bootcamp/assets/12043/0a9dc956-5898-4ce1-89cd-6600d869d508">

 
Click on the watsonx Orchestrate icon on the bottom right and start chatting with your agent.



