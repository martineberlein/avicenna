#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        for i in set(S):
            if i in ['a','e','i','o','u','A','E','I','O','U']:
                S=S.replace(i,'')
        return S


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