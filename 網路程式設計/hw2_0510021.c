#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<unistd.h>
#include<sys/socket.h>
#include<string.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<json/json.h>
#include<memory.h>
struct login_member{
	char id[100];
	char token[100];
	int login;
	char command_token[100];

}mem[100];
int sockfd;
void create_socket(){
	sockfd = 0;
	sockfd = socket(AF_INET, SOCK_STREAM,0);
	if(sockfd == -1){
		printf("fail to create socket\n");
		exit(1);
	}



}
int main(){
        char ip[100];
        char port[100];
	char in;
	int tmp = 0;
	int num_user = 0;
	while((in = getchar())!='\n'){
		ip[tmp++] = in;
	}
	ip[tmp] = '\0';
	tmp = 0;
	while((in = getchar())!='\n'){
		port[tmp++] = in;	
	}
	port[tmp] = '\0';
	
	

	
	while(1){
		create_socket();

		// socket connection info
		struct sockaddr_in info;
		bzero(&info,sizeof(info));
		info.sin_family = AF_INET;

		// local host test
		inet_pton(AF_INET,ip,&info.sin_addr);
		//info.sin_port = htons(atoi(port));
	 	info.sin_port = htons(atoi(port));


		// parameter
		
		char input[1000];
		char msg_buf[1000];
		char* command;
		char tmp_command[1000];
		char* delim = " \n";
		char* pch;
		char *args[40];
		int j = 0;
		json_object *status = json_object_new_object();
		json_object *jobj = json_object_new_object();
		json_object *message = json_object_new_object();
		json_object *jtoken = json_object_new_object();
		//input command

		printf("command-> ");
		fgets(input,sizeof(input),stdin);
		strcpy(tmp_command,input);
		command = strtok(input,"\n");
		pch = strtok(tmp_command,delim);
		while(pch!=NULL){
			args[j] = pch;
			pch = strtok(NULL,delim);
			j++;
		}

		//command with token
		// delete logout list-invite list-friend receive-post
		int idx = -1;
		/*if){
			
			for(int i = 0; i<strlen(command);i++){
				command[i] = 0;
			}
			for(int i = 0; i<num_user;i++){
				if(strcmp(mem[i].id,args[1])==0){
					idx = i;
					break;
				}				
			
			}
			if(idx != -1)	{
				strcat(command,args[0]);
				strcat(command," ");
				strcat(command,mem[idx].token);
				strcat(command,"\0");
			}
			else{	
				strcat(command,args[0]);
				strcat(command," ");
			}
			printf("send command: %s\n",command);
		}	*/
		// invite  accept-invite post
		if(strcmp(args[0],"invite")==0 || strcmp(args[0],"accept-invite")==0 || strcmp(args[0],"post")==0||(strcmp(args[0],"delete")==0 || strcmp(args[0],"logout")==0 || strcmp(args[0],"list-invite")==0 || strcmp(args[0],"list-friend")==0 || strcmp(args[0],"receive-post")==0)){
			for(int i = 0; i<strlen(command);i++){
				command[i] = '\0';
			}
			for(int i = 0; i<num_user;i++){
				if(strcmp(mem[i].id,args[1])==0){
					idx = i;
					break;
				}				
			
			}
			if(idx != -1)	{
				strcat(command,args[0]);
				strcat(command," ");
				strcat(command,mem[idx].token);
				for(int i = 2; i<j; i++){
					strcat(command," ");
					strcat(command,args[i]);
				}
				
			}
			else{
				strcat(command,args[0]);
				for(int i = 2; i<j; i++){
					strcat(command," ");
					strcat(command,args[i]);
				}
			}
			//printf("send command: %s\n",command);
		}

		// exit command

		if(strcmp(args[0],"exit")==0){
			break;
		}

		// other commands

		connect(sockfd,(struct sockaddr*)&info,sizeof(info));
		write(sockfd,command,strlen(command));
		int bytes = 1;
		bytes = read(sockfd,msg_buf,1000);
		msg_buf[bytes] = 0;
		//printf("size of buf = %lu, READ: %s\n",strlen(msg_buf),msg_buf);
		jobj = json_tokener_parse(msg_buf);
		//printf("load: %s\n",json_object_to_json_string(jobj));
		json_object_object_get_ex(jobj,"status",&status);
		
		//command register fin

		if(strcmp(args[0],"register")==0){
			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}
		//command login fin 

		if(strcmp(args[0],"login")==0){
			if(atoi(json_object_to_json_string(status))==0){
				char* tmp_s;
				char tmp_token[1000];
				json_object_object_get_ex(jobj,"token",&jtoken);
				strcpy(tmp_token,json_object_to_json_string(jtoken));
				tmp_s = strtok(tmp_token,"\"");
				strcpy(mem[num_user].token,tmp_s);	
				strcpy(mem[num_user].id,args[1]);
				//debug log
				
				//printf("user %d,id = %s,token = %s\n",num_user,mem[num_user].id,mem[num_user].token);

				num_user++;
			}
			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}

		//command delete fin

		if(strcmp(args[0],"delete")==0){
			if(atoi(json_object_to_json_string(status))==0){
				memset(&mem[idx],0,sizeof(struct login_member));
			}

			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}
		//command logout fin
		if(strcmp(args[0],"logout")==0){
			if(atoi(json_object_to_json_string(status))==0){
				memset(&mem[idx],0,sizeof(struct login_member));
			}
			for(int i = 0;i<strlen(command);i++){
				command[i] = '\0';			
			}
			json_object_object_get_ex(jobj,"message",&message);
			printf("logout: %s\n",json_object_get_string(message));
		}
		//command list-invite fin
		if(strcmp(args[0],"list-invite")== 0){
			if(atoi(json_object_to_json_string(status))==1){
				json_object_object_get_ex(jobj,"message",&message);
				printf("%s\n",json_object_get_string(message));
			}
			else{
				/*char* de = "[],";
				char* p;
				char* tmp_arg[50];
				char tmp_str[100];
				int k=0;
				json_object_object_get_ex(jobj,"invite",&message);
				strcpy(tmp_str,json_object_to_json_string(message));
				p = strtok(tmp_str,de);
				while(p!=NULL){
					tmp_arg[k] = p;
					p = strtok(NULL,de);
					k++;
				}
				for(int i = 0; i<k;i++){
					printf("%s\n",tmp_arg[i]);
				}*/
				json_object_object_get_ex(jobj,"invite",&message);
				int k = 0;
				while(json_object_array_get_idx(message,k)!=NULL){
					printf("%s\n",json_object_get_string(json_object_array_get_idx(message,k)));
					k++;
				}
				
			}

		}
		//command invite  fin
		if(strcmp(args[0],"invite")==0){
			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}
		//command list-friend fin
		if(strcmp(args[0],"list-friend")==0){
			if(atoi(json_object_to_json_string(status))==1){
				json_object_object_get_ex(jobj,"message",&message);
				printf("%s\n",json_object_get_string(message));
			}
			else{
				json_object *tmp_o = json_object_new_object();
				json_object_object_get_ex(jobj,"friend",&message);
				int k = 0;
				for(int i = 0; i<json_object_array_length(message);i++){
					tmp_o = json_object_array_get_idx(message,i);
					printf("%s\n",json_object_get_string(tmp_o));
				}
			}
		}
		//command receive-post
		if(strcmp(args[0],"receive-post")==0){
			if(atoi(json_object_to_json_string(status))==1){
				json_object_object_get_ex(jobj,"message",&message);
				printf("%s\n",json_object_get_string(message));
			}
			else{
				json_object_object_get_ex(jobj,"post",&message);
				int k = 0;
				while(json_object_array_get_idx(message,k)!=NULL){
					json_object *tmp = json_object_array_get_idx(message,k);
					json_object *jid = json_object_new_object();
					json_object *jmsg = json_object_new_object();
					json_object_object_get_ex(tmp,"id",&jid);
					json_object_object_get_ex(tmp,"message",&jmsg);
					printf("%s: %s\n",json_object_get_string(jid),json_object_get_string(jmsg));
					k++;
				}
			}
		}
		//command accept-invite fin
		if(strcmp(args[0],"accept-invite")==0){
			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}
		//command post fin
		if(strcmp(args[0],"post")==0){
			json_object_object_get_ex(jobj,"message",&message);
			printf("%s\n",json_object_get_string(message));
		}
	}
        close(sockfd);



    	return 0;





}
