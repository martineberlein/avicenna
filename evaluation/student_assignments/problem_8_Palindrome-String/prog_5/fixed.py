#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        si = 0
        ri = len(S) - 1
        while si < ri:
            if S[si] != S[ri]:
                return 0
            si += 1
            ri -= 1
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