#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        result = ""
        for c in S:
            if c not in "aeiou":
                result += c
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