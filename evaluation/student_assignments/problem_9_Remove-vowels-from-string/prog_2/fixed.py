#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        v=['a','e','i','o','u']
        r=''
        for i in S:
            if i in v:
                pass
            else:
                r+=i
        return r


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