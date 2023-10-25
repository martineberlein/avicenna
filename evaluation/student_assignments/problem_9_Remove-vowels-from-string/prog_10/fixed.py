#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        n = len(S)
        result = ""
        i = 0
        while(i<=n-1):
            if (S[i]!='a' and S[i]!='e' and S[i]!='i' and S[i]!='o' and S[i]!='u'):
                result += S[i]
            i+=1
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