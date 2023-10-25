#User function Template for python3
class Solution:
    def removeVowels(self, S):
        resstring=""
        for i in S:
            if i in "a" or "e" or "i" or "o" or "u"or"A"or"E"or"I"or"O"or"U":
                pass
            else:
                resstring+=i
        return resstring


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