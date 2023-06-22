#include <iostream>
#include <utility>
using namespace std;

struct ListNode {
     int val;
     ListNode *next;
     ListNode() : val(0), next(nullptr) {}
     ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
};

pair<ListNode*, ListNode*> reverse(ListNode *head, ListNode *tail) {
    ListNode *pre = nullptr;
    ListNode *t = head;
    while(t != tail) {
        ListNode *temp = t->next;
        t->next = pre;
        pre = t;
        t = temp;
    }
    t = tail->next;
    tail->next = pre;
    head->next = t;
    return make_pair(tail, head);
}


ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* pre = new ListNode(0);
    pre->next = head;
    ListNode* tail = pre;
    ListNode* temp = pre;
    while(tail->next) {
        for (int i = 0; i < k; ++i) {
            if (tail->next == nullptr) {
                return pre->next;
            }
            tail = tail->next;
        }
        auto res = reverse(head, tail);
        temp->next = res.first;
        temp = res.second;
    }
    return pre->next;
}