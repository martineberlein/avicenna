#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        result = ""
        for i in S:
            if i not in ['a','e','i','o','u']:
                result += i
        return result


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
	T=int(input())
	for i in range(T):
		s = input()
		
		ob = Solution()	
		answer = ob.removeVowels(s)
		
		print(answer)


# } Driver Code Ends