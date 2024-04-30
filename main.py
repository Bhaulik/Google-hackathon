from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser



# Import environment variables securely
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] =  'AIzaSyCUGkPx62RfjV1TtTjDjEXinnr9kWzUYSA' #input("Enter your Google API key: ")
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the API with the secure environment variable
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.4)

# question
def build_prompt(input_code, interview_question):
    template = f"""
    >>>>> Introduction
    You are tasked with evaluating the following Python function based on the interview coding question
    {interview_question}
    . Please provide a detailed analysis.

    >>>>> Provided Code
    ||| {input_code} |||

    >>>>> Evaluation Results
    ----- Execution: Please simulate the function execution and provide outputs for given test cases and whether it fails or passes.
    ----- Time Complexity: State the Big O time e.g. O(1), O(n)
    ----- Space Complexity: State the Big O time e.g. O n(log n), O(n)
    ----- Optimization: Suggest any potential improvements in code efficiency.

    >>>>> Errors and Warnings
    ----- List any syntax errors or logical mistakes.
    ----- Highlight possible exceptions and handle them in the provided code.

    >>>>> Conclusion
    Summarize the overall code quality and provide final recommendations.

    >>>>> End of Analysis
    
    For example: 
**Introduction:**

The provided Python function, `twoSums`, is designed to find two elements in a given list `nums` that sum up to a specified `target`.

**Execution:**
**Question:**
Two Sum:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

**Test Case 1:**
Input: `nums = [2, 7, 11, 15]`, `target = 9`
Output: `[0, 1]` (Indices of elements 2 and 7)
test case pass

**Test Case 2:**
Input: `nums = [3, 2, 4]`, `target = 6`
Output: `[1, 2]` (Indices of elements 2 and 4)
test case pass

**Test Case 3:**
Input: `nums = [3, 3]`, `target = 6`
Output: `[0, 1]` (Indices of both elements 3)
test case pass


**Optimization:**

The code can be optimized in terms of efficiency:

* **Use a Set Instead of a Dictionary:** The `complements` dictionary can be replaced with a set, which offers faster lookup and insertion.
* **Early Return:** After finding the pair of numbers that sum up to the target, the function can return immediately instead of iterating through the entire list.

**Errors and Warnings:**

* **Logical Error:** The code may not handle cases where the `target` is less than the smallest number in `nums`.
* **Exception Handling:** The code does not handle the case where the target sum is not found.

**Conclusion:**

Overall, the code is functionally correct but can be improved in terms of efficiency and exception handling.

**Recommendations:**

* **Optimization:** Implement the suggested optimizations to enhance code performance.
* **Error Handling:** Add an exception handler to handle the case where the target sum is not found.
* **Documentation:** Provide clear documentation explaining the purpose of the function and any assumptions or limitations.

**Improved Code:**

```python
def twoSums(self, nums: List[int], target: int) -> List[int]:
    complements = set()
    for i, num in enumerate(nums):
        complement = target - num
        if complement in complements:
            return [complements.pop(complement), i]
        else:
            complements.add(num)
    return []

    """
    return template

@app.get("/")
async def root():
    try:
        instructions = ""
        # Invoke the model and handle errors gracefully
        code = """
        def twoSums(self, nums: List[int], target: int) -> List[int]:
        complements = {}
        for i, num in enumerate(nums):
            complement = target+num
            if num in complements:
                complements[complement] = i + 1
            else:
                return [complements[num], i]
        
        return []
        """
        
        question = """
        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
        """
        q = build_prompt(code, question)
        # print(q)
        
        result = llm.invoke(q)
        # print(result)
        # print(result)
        # print(result.content)
        # Check if the API call was successful
        # if result.status_code == 200:
        
        return {"message": result.content, "input": code}
        # else:
        #     return {"error": "Failed to retrieve content from API", "status": result.status_code}
    except Exception as e:
        return {"error": str(e)}
    

class CodeInput(BaseModel):
    code: str

def codeanalysistemplate (code, format_instructions):
    template = f"""
    You are a code analysis agent during an interview,
    Based on the format instructions: {format_instructions}
    if the code is not valid code or something empty or stupid, return -1 for all values except syntax for which you'll say invalid syntax
    
    Now analyze the code: 
    ------
    {code}
    ------
    """
    return template

def code_analysis_template(code, format_instructions):
    return f"""
    Analyze the following code:
    ```
    {code}
    ```
    Based on the format instructions: {format_instructions}
    Please provide:
    1. The estimated time complexity.
    2. The estimated space complexity.
    3. Any syntax errors.
    4. Whether the code compiles successfully.
    5. A code quality meter on a scale of 1 to 10.
    """

@app.post("/analyze_code/")
async def analyze_code(data: CodeInput):
    print('Starting analysis')
    print('----')
    print(data.code)  # Ensure this prints the expected code string
    print('----')
    
    response_schemas = [
    ResponseSchema(name="time_complexity", description="time complexity notation like O(N)", type="string"),
    ResponseSchema(name="space_complexity", description="space complexity notatino like O(N)", type="string"),
    ResponseSchema(name="syntax_errors", description="any error messages that are caused by the code",type="string"),
    ResponseSchema(name="successful_compilation", description="boolean value which will be false if there are syntax_errors", type="boolean"),
    ResponseSchema(name="code_quality_meter", description="an integer out of 10; base it on code readability, space and time complexity", type="integer") 
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    print('---format Instructions---')
    print(format_instructions)
    print('---format Instructions---')
    formatted_instructions = output_parser.get_format_instructions()
    prompt_text = code_analysis_template(data.code, formatted_instructions)
    print('>>>')
    print(prompt_text)
    print('>>>')

    # prompt = codeanalysistemplate(data.code)  # Pass the actual code to the template generator
    prompt = PromptTemplate(template=prompt_text, input_variables=["question"],
                                partial_variables={"format_instructions": format_instructions})
    try:
        result = llm.invoke(prompt_text)
        print('---result--')
        print(result)
        print('---result--')
        # Assuming result.content holds the necessary data:
        analysis = result.content if result.content else {}

        # Simulated response parsing - adapt based on actual model response
        # time_complexity = analysis.get('time_complexity', 'Unknown')
        # space_complexity = analysis.get('space_complexity', 'Unknown')
        # syntax_errors = analysis.get('syntax_errors', 'None detected')
        # successful_compilation = analysis.get('successful_compilation', False)
        # quality_meter = analysis.get('code_quality_meter', 0)

        return {
            analysis
            # "time_complexity": time_complexity,
            # "space_complexity": space_complexity,
            # "syntax_errors": syntax_errors,
            # "successful_compilation": successful_compilation,
            # "code_quality_meter": quality_meter
        }
    except Exception as e:
        print(str(e))  # Print the error for debugging
        raise HTTPException(status_code=500, detail=str(e))


    # return {
    #     "time_complexity": f"O({time_complexity})" if "O(" not in time_complexity else time_complexity,
    #     "space_complexity": space_complexity,
    #     "syntax_errors": syntax_errors,
    #     "successful_compilation": successful_compilation,
    #     "code_quality_meter": quality_meter
    # }
    
example_gen_q = """
Use the following as a examples of kind of questions to generate, don't generate the same thing again and again.

Question:
------------------------------------
Median of Two Sorted Arrays:
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
The overall run time complexity should be O(log (m+n)).

------------------------------------
Difficulty: Hard

Topics: Array, Binary Search , Divide and Conquer

------------------------------------
Examples: 

Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

------------------------------------
Constraints:

nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-10^6 <= nums1[i], nums2[i] <= 10^6

------------------------------------
TestCases:

Case 1:
nums1 = [1,3]
nums2 = [2]

Case 2:
nums1 = [1,2]
nums2 = [3,4]

------------------------------------
Most optimal Solution: 
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        n1 = len(nums1)
        n2 = len(nums2)
        
        # Ensure nums1 is the smaller array for simplicity
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        
        n = n1 + n2
        left = (n1 + n2 + 1) // 2 # Calculate the left partition size
        low = 0
        high = n1
        
        while low <= high:
            mid1 = (low + high) // 2 # Calculate mid index for nums1
            mid2 = left - mid1 # Calculate mid index for nums2
            
            l1 = float('-inf')
            l2 = float('-inf')
            r1 = float('inf')
            r2 = float('inf')
            
            # Determine values of l1, l2, r1, and r2
            if mid1 < n1:
                r1 = nums1[mid1]
            if mid2 < n2:
                r2 = nums2[mid2]
            if mid1 - 1 >= 0:
                l1 = nums1[mid1 - 1]
            if mid2 - 1 >= 0:
                l2 = nums2[mid2 - 1]
            
            if l1 <= r2 and l2 <= r1:
                # The partition is correct, we found the median
                if n % 2 == 1:
                    return max(l1, l2)
                else:
                    return (max(l1, l2) + min(r1, r2)) / 2.0
            elif l1 > r2:
                # Move towards the left side of nums1
                high = mid1 - 1
            else:
                # Move towards the right side of nums1
                low = mid1 + 1
        
        return 0 # If the code reaches here, the input arrays were not sorted.

------------------------------------
"""
example_gen_q2= """
Example 2:
------------------------------------
Two Sum:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

------------------------------------
Difficulty: Easy

Topics: Array, Hash Table

------------------------------------
Examples: 

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

------------------------------------
Constraints:

2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
-10^9 <= target <= 10^9

------------------------------------
TestCases:

Case 1:
nums = [2,7,11,15]
target = 9

Case 2:
nums = [3,2,4]
target = 6

Case 3:
nums = [3,3]
target = 6

------------------------------------
Most optimal Solution: 
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        complements = {{}}
        for i, num in enumerate(nums):
            complement = target-num
            if num not in complements:
                complements[complement] = i+1
            else:
                return [complements[num], i]
        return []

------------------------------------
"""

class QuestionGenerateModel(BaseModel):
    difficulty: str
    ds: str

class GeneratedQuestionModel(BaseModel):
    problem: str
    difficulty: str
    topics: List[str]
    examples: List[str]
    constraints: str
    test_cases: List[str]
    optimal_solution: str
     

#could add a programming language selection as a param
@app.post("/generate_leetcode_question_breakdown")
async def generate_leetcode_question_breakdown(model: QuestionGenerateModel):
    template = """
    Create a detailed LeetCode-style which is considered a {difficulty} problem/challenge - ensure the problem statement is viable.
    The idea solution could use at least one  
    {ds} data structure(s). Based on this, generate a problem challenge which as
            examples, constraints, test cases, solution (ensure that the solution is accurate) similar to the follow examples: 
            Example:
            """ + example_gen_q + """another example you can get inspiration from is """+ example_gen_q2 + """
            since I would need to parse the output in json in my frontend, after you generate the coding challenge
            convert your new generated question to JSON with the following 
            keys: 
            problem: str
            difficulty: str
            topics: List[str]
            examples: List[str]
            constraints: str
            test_cases: List[str]
            optimal_solution: str
            """

    prompt = PromptTemplate.from_template(template)
    print('the PROMPT IS ')
    print(prompt)
    # prompt = PromptTemplate(template=template, input_variables=["difficulty", "ds"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm
    # response = chain.invoke({"difficulty": model.easy, 'ds': model.ds})
    response = chain.invoke({"difficulty": model.difficulty, "ds": model.ds})
    # print('the response is : ')
    # print(response)
    return response

###
class PromptData(BaseModel):
    question: str

example_q1 = """
------------------------------------
Median of Two Sorted Arrays:
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
The overall run time complexity should be O(log (m+n)).

------------------------------------
Difficulty: Hard

Topics: Array, Binary Search , Divide and Conquer

------------------------------------
Examples: 

Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

------------------------------------
Constraints:

nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-10^6 <= nums1[i], nums2[i] <= 10^6

------------------------------------
TestCases:

Case 1:
nums1 = [1,3]
nums2 = [2]

Case 2:
nums1 = [1,2]
nums2 = [3,4]

------------------------------------
Most optimal Solution: 
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        n1 = len(nums1)
        n2 = len(nums2)
        
        # Ensure nums1 is the smaller array for simplicity
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        
        n = n1 + n2
        left = (n1 + n2 + 1) // 2 # Calculate the left partition size
        low = 0
        high = n1
        
        while low <= high:
            mid1 = (low + high) // 2 # Calculate mid index for nums1
            mid2 = left - mid1 # Calculate mid index for nums2
            
            l1 = float('-inf')
            l2 = float('-inf')
            r1 = float('inf')
            r2 = float('inf')
            
            # Determine values of l1, l2, r1, and r2
            if mid1 < n1:
                r1 = nums1[mid1]
            if mid2 < n2:
                r2 = nums2[mid2]
            if mid1 - 1 >= 0:
                l1 = nums1[mid1 - 1]
            if mid2 - 1 >= 0:
                l2 = nums2[mid2 - 1]
            
            if l1 <= r2 and l2 <= r1:
                # The partition is correct, we found the median
                if n % 2 == 1:
                    return max(l1, l2)
                else:
                    return (max(l1, l2) + min(r1, r2)) / 2.0
            elif l1 > r2:
                # Move towards the left side of nums1
                high = mid1 - 1
            else:
                # Move towards the right side of nums1
                low = mid1 + 1
        
        return 0 # If the code reaches here, the input arrays were not sorted.

------------------------------------
"""
example_q2= """
------------------------------------
Two Sum:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

------------------------------------
Difficulty: Easy

Topics: Array, Hash Table

------------------------------------
Examples: 

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

------------------------------------
Constraints:

2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
-10^9 <= target <= 10^9

------------------------------------
TestCases:

Case 1:
nums = [2,7,11,15]
target = 9

Case 2:
nums = [3,2,4]
target = 6

Case 3:
nums = [3,3]
target = 6

------------------------------------
Most optimal Solution: 
class Solution {
def twoSum(self, nums: List[int], target: int) -> List[int]:
        complements = {}
        for i, num in enumerate(nums):
            complement = target-num
            if num not in complements:
                complements[complement] = i+1
            else:
                return [complements[num], i]
        return []

------------------------------------
"""

    
@app.post("/generate_leetcode_question")
async def generate_leetcode_question(data: PromptData):
    print('Generating Leetcode Question')
    # create our examples
    examples = [
        {
            "query": """Create a detailed LeetCode-style which is considered a hard problem statement based on this idea, provide 
            examples, constraints, test cases, solution """,
            "answer": example_q1
        }, {
            "query": """Create a detailed LeetCode-style which is considered an easy problem statement based on this idea, provide 
            examples, constraints, test cases, solution """,
            "answer": example_q2
        }
    ]
    
    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """
    
    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )
    
    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """You are an interview expert, 
    create a detailed LeetCode-style based on the difficulty. Provide 
    examples, constraints, test cases, solutions. Here are some examples:
    """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """
    

        
    try:
        # Construct the prompt for generating a LeetCode question
        # full_prompt = f"Create a detailed LeetCode-style problem statement based on this idea: {data.prompt}"
        # # Generate the question using LangChain
        # prompt_template_name = PromptTemplate(input_variables=['difficulty', 'data_structure']),
        # template = "Create a detailed LeetCode-style problem statement based on this idea"
        # now create the few shot prompt template
        few_shot_prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            example_separator=""
        )
        print('Few Shot Prompt Template is ')
        print(few_shot_prompt_template)
        query = """Create a detailed LeetCode-style which is considered a hard problem statement based on this idea, provide 
            examples, constraints, test cases, solution"""
        result_generated = few_shot_prompt_template.format(query=query)
        print(result_generated)
        print('result is : -->')
        # resultofapi = llm.invoke(result_generated)
        # print(resultofapi)
        
        return {"generated_question": 'result_generated'}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))