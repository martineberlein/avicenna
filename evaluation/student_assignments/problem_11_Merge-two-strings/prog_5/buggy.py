#User function Template for python3
class Solution:
    def merge(self, S1, S2):
        # Initialize the result string
        result = ""

        # Iterate over the characters of both strings
        for c1, c2 in zip(S1, S2):
            result += c1 + c2

        # Append the remaining characters of S1 if S2 is shorter
        result += S1[len(S2):]

        # Return the result string
        return result


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        S1,S2 = map(str,input().strip().split())
        ob = Solution()
        print(ob.merge(S1, S2))
# } Driver Code Ends