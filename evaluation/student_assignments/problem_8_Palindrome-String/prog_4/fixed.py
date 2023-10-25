#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        i = 0
        j = len(S) - 1
        # code here
        if(len(S) == 1):
            return 1
        while(i<j):
            if(S[i] != S[j]):
                return 0
            i += 1
            j -= 1
        return 1


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
	T=int(input())
	for i in range(T):
		S = input()
		ob = Solution()
		answer = ob.isPalindrome(S)
		print(answer)

# } Driver Code Ends