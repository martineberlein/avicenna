#User function Template for python3
def check(s, si, ri):
    if si >= ri:
        return 1
    if s[si] == s[ri]:
        check(s, si+1, ri-1)
    return 0
    
class Solution:
    def isPalindrome(self, S):
        return check(S, 0, len(S)-1)



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