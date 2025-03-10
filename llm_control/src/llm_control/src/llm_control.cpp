#include <ros/ros.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

int control = 0;

struct CommandHead{ 
    uint32_t code;
    uint32_t paramters_size;
    uint32_t type;
}; 

void controlHandler(const std_msgs::Int32::ConstPtr& flag) {
    control = flag->data;
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "llm_control");
    ros::NodeHandle nh;

    ros::Subscriber subControl = nh.subscribe<std_msgs::Int32> ("/llm_control", 1, controlHandler);
    
    const char *server_ip = "192.168.1.120";
    const int server_port = 43893;
    
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        close(sockfd);
        return 1;
    }

    ros::Rate rate(1);
    bool status = ros::ok();

    while (status) {

        control = 0;
        ros::spinOnce();

        struct CommandHead command_head = {0};
        if (control == 6) {
            command_head.code = 0x2101030C; 
        }
        else if(control == 7){
            command_head.code = 0x21010506;
        }
        else if(control == 8){
            command_head.code = 0x21010204;
        }
        else {
            continue;
        }

        command_head.paramters_size = 0;
        command_head.type = 0;

        ssize_t sent_bytes = sendto(sockfd, &command_head, sizeof(command_head), 0,
            (struct sockaddr*)&server_addr, sizeof(server_addr));
        if (sent_bytes < 0) {
            std::cerr << "Error sending message" << std::endl;
            close(sockfd);
            return 1;
        }


    }

    close(sockfd);
    return 0;

}
