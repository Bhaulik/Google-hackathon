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
    os.environ["GOOGLE_API_KEY"] =  ' enter api key ' #input("Enter your Google API key: ")
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
    3. Whether the code compiles successfully.
    4. A code quality meter on a scale of 1 to 10.
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
    ResponseSchema(name="successful_compilation", description="boolean value which will be false if there are syntax_errors", type="boolean"),
    ResponseSchema(name="code_quality_meter", description="an integer out of 10; base it on code readability, space and time complexity", type="integer") 
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    formatted_instructions = output_parser.get_format_instructions()
    print('---format Instructions---')
    print(formatted_instructions)
    print('---format Instructions---')
    prompt_text = code_analysis_template(data.code, formatted_instructions)
    print('>>>')
    print(prompt_text)
    print('>>>')

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
    {ds} data structure(s). Based on this, generate a problem challenge which has
            examples, constraints, test cases, solution (ensure that the solution is accurate) similar to the follow examples: 
            Example:
            """ + example_gen_q + """another example you can get inspiration from is """+ example_gen_q2 + """
            since I would need to parse the output in json in my frontend, after you generate the coding challenge
            convert your new generated question to JSON with the following. Ensure you escape the strings properly and that
            the output will work seemlessly with JSON.parse function without any errors or exceptions when used against the output. 
            keys: 
            problem: str
            difficulty: str
            topics: List[str] //the elements should strictly be strings not objects
            constraints: str
            test_cases: List[str] //the elements should strictly be strings and not objects
            optimal_solution: str
            """

    prompt = PromptTemplate.from_template(template)
    print('the PROMPT IS ')
    print(prompt)
    chain = prompt | llm
    response = chain.invoke({"difficulty": model.difficulty, "ds": model.ds})

    return response

spare_string = "---------SetApproachclassSolution{public:boolcontainsDuplicate(vector<int>&nums){returnnums.size()-50000>set<int>(nums.begin(),nums.end()).size();}} ---------"
spare_string2 = "---------classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=100;i<nums.length;++i){if(nums[i]==nums[i-1]){returntrue;}}returnfalse;}}---------"
class CodeRunModel(BaseModel):
    code: str
    question: str
    test_cases: str
    constraints: str

@app.post("/runcode")
async def runcode(code_to_run: CodeRunModel):
    print(code_to_run)
    try:
        # Invoke the model and handle errors gracefully
        code = """
        {code_to_run}
        """
        
        question = """
        {question}
                """
        q = build_prompt(code_to_run.code, code_to_run.question)
        # print(q)
        
        result = llm.invoke(q)
        print(result)
        
        return {"message": result.content, "input": code}
        # else:
        #     return {"error": "Failed to retrieve content from API", "status": result.status_code}
    except Exception as e:
        return {"error": str(e)}
    
    
    
def code_run_template(question, constraints, code, format_instructions):
    return f"""
    As an expert and a person who gets all interview programming questions right and an interviewer.
    Consider the following interview coding question that is asked to an interviewee:
    ```
    QUESTION: 
    {question} 
    
    CONSTRAINTS:
    Constraints are : {constraints}
    ```
    Now,
    Identify at least 100 test cases for the above question and figure out correctly the answers to them.
    Analyze and accurately run and verify following code written by the interviewee for the coding round
    for all those test cases and if any of them fails then the solution provided by the interviewee is wrong.
    :
    ```
    The code written by the interviewee is:
    CODE TO VERIFY is between the --------- characters below
    ---------
    {code}
    ---------
    ```
    Ignore the name of the function in {code} First figure out the right code answer to our {question} then compare the run test results of
    {code} TO VERIFY that the outputs/test cases expected results matches the answers to our {question}. If the 
    optimal solution to {question} does not match the {code} then solution by interviewee to our question is wrong!
    
    Analyze the interviewee {question} (if there is a syntax or logic error then we are done, set right_solution as false) and provide:
    1. The time complexity of the solution by the interviewee .
    2. The estimated space complexity of solution by the interviewee.
    3. Errors (true or false value) of solution by the interviewee
    4. A code quality meter on a scale of 1 to 10 of solution by the interviewee.(use metrics like clean code and functional code, readability etc.)
    5. Code report - showing step by step how the proble could be solved
    6. The optimal code solution to the problem, always write the CODE!! which has the best time and space complexity in the same programming language the interviewee used to code
    7. Figure out the test cases the solution will fail at, if it succeed in all then say that it successfully passed all test cases
    8. Programming language of the interviewee based on the code the interviewee wrote
    9. Whether the solution provided by the interviewee is wrong or right by comparing the solution that you came up with and the one that the interviewee inputted.

    return in json format with only the following 10 keys and matching to the above:
    time_complexity - undefined if you can't determine or if the solution is wrong
    space_complexity- undefined if you can't determine
    errors- undefined if you can't determine
    code_quality_meter- undefined if you can't determine
    code_report- undefined if you can't determine
    optimal_solution- undefined if you can't determine
    test_case_output- undefined if you can't determine
    programming_language - undefined if its invalid code
    right_solution - false if the solution provided by the interviewee to the question is wrong
    test_cases_used - shows at least 5 test cases that were ran against the solution, if there are any test cases that would fail for this scenario bring them up please please
    , show the input, output and expected value with a ✅ if pass or ❌ if failed the test case
    
    PLEASE ENSURE THAT THE SOLUTION PROVIDED IS VERIFIED CORRECTLY!!!! IF ITS WRONG DO UPDATE THE OUTPUT MENTIONED ABOVE CORRECTLY!!!
    
   USE AS AN EXAMPLE THE FOLLOWING SCENARIO:
   
    QUESTION
    Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

    CODE TO VERIFY
    def containsDuplicate(self, nums: List[int]) -> bool:
        distinct = set()
        for n in nums:
            if n in distinct:
                return False
            distinct.add(n)
        return False
     for the constraints "1 <= nums.length <= 10^5-1^9 <= nums[i] <= 10^9",
     
 THE RESULT AFTER EVALUATION WILL BE : 
  "time_complexity": "undefined",
  "space_complexity": "undefined",
  "errors": true,
  "code_quality_meter": 1,
  "code_report": "You have a logic issue in your code on the first line : return False",
  "optimal_solution": "<write code for the optimal solution>",
  "test_case_output": "Test Case failed for input: [1,2,3,1], the output was false but expected true",
  "programming_language": "python",
  "right_solution": false
  "test_cases_used": <show 5 test cases with input, output and expected> alue with a ✅ if pass or ❌ if failed the test case
  
  Example 2:
  
  QUESTION
  Two Sum
  
  CODE TO VERIFY """ + spare_string + """
    THE RESULT AFTER EVALUATION WILL BE : 
    "time_complexity": "undefined",
    "space_complexity": "undefined",
    "errors": true,
    "code_quality_meter": 1,
    "code_report": "You have a logic issue in your code",
    "optimal_solution": "<write code for the optimal solution>",
    "test_case_output": "<a lot of test cases fail, indicate max of 3 test cases that might fail>",
    "programming_language": "cpp",
    "right_solution": false  
    "test_cases_used": <show 5 test cases with input, output and expected> alue with a ✅ if pass or ❌ if failed the test case
    
Example 2:
  
  QUESTION
  Two Sum
  
  CODE TO VERIFY """ + spare_string2 + """
    THE RESULT AFTER EVALUATION WILL BE : 
    "time_complexity": "undefined",
    "space_complexity": "undefined",
    "errors": true,
    "code_quality_meter": 3,
    "code_report": "You have a logic issue in your code",
    "optimal_solution": "classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=1;i<nums.length;++i){if(nums[i]==nums[i-1]){returntrue;}}returnfalse;}}",
    "test_case_output": "<a lot of test cases fail, indicate max of 3 test cases that might fail>",
    "programming_language": "java",
    "right_solution": false
    "test_cases_used": <show 5 test cases with input, output and expected> alue with a ✅ if pass or ❌ if failed the test case
    
    Example 3:
    Two Sum
    CODE TO VERIFY 
    ---------
    classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=1;i<nums.length-1000;++i){if(nums[i-2]==nums[i-1]){returntrue;}}returnfalse;}}
    ---------
    THE RESULT AFTER EVALUATION WILL BE : 
    "time_complexity": "undefined",
    "space_complexity": "undefined",
    "errors": true,
    "code_quality_meter": 0,
    "code_report": "You have a logic issue in your code, you are using length-1000, so its doesn't passtest cases, also nums[i-2] bring about index out of bounds",
    "optimal_solution": "classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=1;i<nums.length;++i){if(nums[i]==nums[i-1]){returntrue;}}returnfalse;}}",
    "test_case_output": "failed test cases",
    "programming_language": "java",
    "right_solution": false
    "test_cases_used": <show 5 test cases with input, output and expected> alue with a ✅ if pass or ❌ if failed the test case
    
    Example 4:
    Minimum Cost to Connect Sticks
    CODE TO VERIFY 
    ---------
    classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=1;i<nums.length-1000;++i){if(nums[i-2]==nums[i-1]){returntrue;}}returnfalse;}}
    ---------
    THE RESULT AFTER EVALUATION WILL BE : 
    "time_complexity": "undefined",
    "space_complexity": "undefined",
    "errors": true,
    "code_quality_meter": 1,
    "code_report": "You have some syntax errors, review code and fix them",
    "optimal_solution": "classSolution{publicbooleancontainsDuplicate(int[]nums){Arrays.sort(nums);for(inti=1;i<nums.length;++i){if(nums[i]==nums[i-1]){returntrue;}}returnfalse;}}",
    "test_case_output": "unable to run",
    "programming_language": "java",
    "right_solution": false
    "test_cases_used": You've got syntax errors, cannot run your code when it has syntax issues and incorrect  heappop() 
    
    Example 5:
    Reverse Linked List II
    CODE TO VERIFY 
    ---------
    public ListNode reverseBetween(ListNode head, int left, int right) {
    ---------
    THE RESULT AFTER EVALUATION WILL BE : 
    "time_complexity": "undefined",
    "space_complexity": "undefined",
    "errors": true,
    "code_quality_meter": 1,
    "code_report": "You have some syntax errors, review code and fix them",
    "optimal_solution": "figure out the optimal solution",
    "test_case_output": "unable to run",
    "programming_language": "java",
    "right_solution": false,
    "test_cases_used": You've got syntax errors, cannot run your code 
    
    ALSO, since I would need to parse the output in json in my frontend javascript, for the result
            convert the output to JSON with the following. Ensure you escape the strings properly and that
            the output will work seemlessly with JSON.parse function without any errors or exceptions when used against the output. 
        output:
        # The output should be a Markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }
        # ```
    """

class CodeExecuteModel(BaseModel):
    question: str
    constraints: str
    code: str
    

@app.post("/code_execute/")
async def code_execute(data: CodeExecuteModel):
    print(data)
    print('executing code analysis')
    print('---------')
    response_schemas = [
    # ResponseSchema(name="time_complexity", description="time complexity notation like O(N)", type="string"),
    # ResponseSchema(name="space_complexity", description="space complexity notation like O(N)", type="string"),
    # ResponseSchema(name="errors", description="boolean value which will be true if there are syntax_errors but if no errors then its false", type="boolean"),
    # ResponseSchema(name="code_quality_meter", description="an integer out of 10; base it on code readability, space and time complexity", type="integer"),
    # ResponseSchema(name="code_report", description="explain any improvements that could have been made to the code that the user executed", type="string"),
    # ResponseSchema(name="optimal_solutionzzz", description="generate the most optimal space and time complexity solution and write CODE for it, ensure you are right!!", type="string"),
    # ResponseSchema(name="test_case_output", description="if all test cases pass state all test cases passed successfully else say failed test case : <give details on the failed test case, input, output and expected value>, consider the constraints provided initially, ensure you get this right!!!", type="string")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    formatted_instructions = output_parser.get_format_instructions(only_json=True)
    prompt_text = code_run_template(data.question, data.constraints, data.code, formatted_instructions)
    print('>>>The prompt is')
    print(prompt_text)
    print('>>>')
    llm2 = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7)
    try:
        result = llm2.invoke(prompt_text)
        print('---result of the LLM is --')
        print(result)
        print('---result--')
        # Assuming result.content holds the necessary data:
        analysis = result.content if result.content else {}

        return analysis
        
    except Exception as e:
        print(str(e))  # Print the error for debugging
        raise HTTPException(status_code=500, detail=str(e))
