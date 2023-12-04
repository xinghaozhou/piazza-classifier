#include <iostream>
#include <fstream>
#include <set>
#include <math.h> 
#include "csvstream.hpp"
using namespace std;

class classifier{
    public:
      classifier(string tr, string te, bool db):p_train(tr), p_test(te), debug(db){};

      int num_post_contain_c(const string &C){
        return label_map[C];
      };
      
      int num_post_contain_w(const string &W){
        return word_map[W];
      };

      int num_post_with_c_contain_w(const string &C, const string &W){
        return cw_map[std::pair<string,string>(C,W)];
      }

      double log_prob_score_label(const string &C){
        return log((float)(num_post_contain_c(C))/((float)(num_post)));
      };

      double log_prob_score_label_word(const string &C, const string &W){
        if(num_post_contain_w(W) == 0){
            return log(1/((float)(num_post)));
        }else if(num_post_with_c_contain_w(C,W) == 0 && num_post_contain_w(W) != 0){
            return log((float)(num_post_contain_w(W))/(float)(num_post));
        }else{
            return log((float)(num_post_with_c_contain_w(C,W))/num_post_contain_c(C));
        }  
      };

      // EFFECTS: Return a set of unique whitespace delimited words.x
      set<string> unique_words(const string &str) {
        istringstream source(str);
        set<string> words;
        string word;
        while (source >> word) {
            words.insert(word);
        }
        return words;
    }
      void process(){
        if(debug){
             cout<<"training data:"<<"\n";
        }
        csvstream csvtrain(p_train);
        map<string, string> row_train;
        while (csvtrain>> row_train) { //load in the entire row
          num_post++; //each time, increase the post++
          if(debug){
            cout <<"  label = " <<  row_train["tag"] << ", ";
            cout << "content = " << row_train["content"] << "\n";
          }
          label_map[row_train["tag"]] +=1;
           // get rid of those repeated words
           set<string> bags_of_words = unique_words(row_train["content"]); 
           for(auto &words:bags_of_words){
            cw_map[std::pair<string, string>(row_train["tag"], words)] += 1;
            word_map[words]+=1; // add words into vocab_set
          }
        }
        vocab_size = word_map.size(); // num of vocab = num of words in bags_of
        cout<<"trained on " << num_post << " examples"<<"\n";
        
        if(debug){
          cout << "vocabulary size = " << vocab_size<<"\n";
          cout<<"\n";
          cout<<"classes:"<<"\n";

        for(auto &label:label_map){
            cout<<"  "<<label.first<<", "<<label.second<<" examples, ";
            cout<<"log-prior = " << log_prob_score_label(label.first)<<"\n";
        }
        cout<<"classifier parameters:"<<"\n";
        for(auto &label:label_map){
            for(auto &word:word_map){
              if(cw_map[pair<string,string>(label.first,word.first)] >= 1){
                cout<<"  "<<label.first <<":"<<word.first<<", count = ";
                cout<<cw_map[pair<string,string>(label.first,word.first)];
                cout<<", ";
                cout<<"log-likelihood = ";
                cout<< log_prob_score_label_word(label.first, word.first)<<"\n";
              }
            }
          }
        }
        cout<<"\n";
        cout<<"test data:"<<"\n";
        csvstream csvtest(p_test);
        map<string, string> row_test;
        //load in the entire row
        while (csvtest>> row_test) { 
            num_test++;
            string max_label; // store max-label
            float max_prob = -INFINITY; // store max-log
            cout<<"  correct = "<<row_test["tag"]<<", ";
            //Get rid of repeated words
            set<string> bags_of_words = unique_words(row_test["content"]);
            for(auto &label:label_map){
                // store current label
                string current_label =  label.first;
                // store current log-prob
                float current_prob = log_prob_score_label(current_label);
                for(auto &word:bags_of_words){
                    // Calculate log-prob for word
                    current_prob += log_prob_score_label_word(label.first, word); 
                }
                if(current_prob > max_prob){ // 
                    max_prob = current_prob;
                    max_label = current_label;
                }
            }
            if(max_label == row_test["tag"]){
                accuracy+=1;
            }
            cout<< "predicted = "<<max_label;
            cout<<", log-probability score = " <<max_prob<<"\n";
            cout<< "  content = "<<row_test["content"] <<"\n"<<endl;
           }
            cout<<"performance: "<<accuracy<<" / "<<num_test;
            cout<<" posts predicted correctly"<<"\n";
        };

    private:
        string p_train;
        string p_test;
        bool debug;
        int num_post = 0;
        int vocab_size = 0;
        int accuracy = 0;
        int num_test = 0;
    
        map<string, int> label_map;
        map<string, int> word_map;
        map<std::pair<string, string>, int> cw_map;

};

int main(int argc, char *argv[]) {
  cout.precision(3);
  string train_file;
  string test_file;
  bool debug = false;

  if(argc == 3){
    train_file = argv[1];
    test_file = argv[2];
    classifier(train_file, test_file, debug).process();
    return 1;
  }else if(argc == 4){
    train_file = argv[1];
    test_file = argv[2];
    if(static_cast<string>(argv[3]) != "--debug"){
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
    }else{
        debug = true;
        classifier(train_file, test_file, debug).process();
        return 1;
    }
   }else{
    cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
    return 1;
  }
    
   

  
}