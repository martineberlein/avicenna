#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # code here
        if len(S)==1:
            return 1
        if len(S)==2:
            if S[0]==S[1]:
                return 1
            elif S[0]!=S[1]:
                return 0
        si=0
        ei=len(S)-1
        while si<ei:
            if S[si]==S[ei]:
                si+=1
                ei-=1
            elif S[si]!=S[ei]:
                return 0
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