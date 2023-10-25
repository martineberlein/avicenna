#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        x=""
        a=["a","e","i","o","u"]
        for i in range(len(S)):
            if S[i] in a:
                x=S.replace(S[i],"")
        return x


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