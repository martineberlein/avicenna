#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # code here
        L, R = 0, len(S)-1
        while L < R:
            if S[L] == S[R]:
                L += 1
                R -= 1
            else:
                return False
        return True


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