import os
import requests
import cv2
import sys
import numpy as np
import base64
from ultralytics import YOLO
import dashscope
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import torch
import pyaudio
import wave
import time
import pyrealsense2 as rs

dashscope.api_key=""  #请输入您的api key

object_name = "None"
robot_x = 0
robot_y = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000

#os.close(sys.stderr.fileno())

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

    img_color = np.asanyarray(aligned_color_frame.get_data())
    img_depth = np.asanyarray(aligned_depth_frame.get_data())

    return depth_intrin, img_color, img_depth

def audio_record(rec_time, filename):
    if rec_time <= 0:
        raise ValueError("Recording time must be positive.")
    if not filename.endswith('.wav'):
        raise ValueError("Filename must end with '.wav'")

    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print('开始识别...')
        start_time = time.time()
        frames = []

        while time.time() - start_time < rec_time:
            data = stream.read(CHUNK)
            frames.append(data)

        print('识别结束。')

        stream.stop_stream()
        stream.close()

        with wave.open(filename, 'wb') as f:
            f.setnchannels(CHANNELS)
            f.setsampwidth(p.get_sample_size(FORMAT))
            f.setframerate(RATE)
            f.writeframes(b''.join(frames))
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        p.terminate()

    #print(f"Audio recording time: {time.time() - start_time:.2f} seconds")

def whisper_transcribe(file_path):
    start_time = time.time()
    try:
        messages = [
        {
            "role": "user",
            "content": [
                {"audio": file_path},
                {"text": "把这段音频转化为文字，不需要别的输出"
                }
            ]
        }
        ]
        response = dashscope.MultiModalConversation.call(model='qwen-audio-chat',
                                                     messages=messages)
        user_prompt = response.output.choices[0].message.content[0]['text'].strip('好的，这是这段音频转化为的文字：')
        print(f"语音转换时长: {time.time() - start_time:.2f} s")
        #print(f"Whisper transcription time: {time.time() - start_time:.2f} seconds")
        return user_prompt
    except Exception as e:
        rospy.logerr(f"Error transcribing audio: {e}")
        #print(f"Whisper transcription failed: {time.time() - start_time:.2f} seconds")
        return None


if __name__ == '__main__':

    rospy.init_node('object_depth_publisher', anonymous=True)

    # 加载 YOLO 模型
    model = YOLO("YOLO/yolov8n.pt") #可以替换为其它yolo模型
    model.conf=0.35

    # 定义ROS发布器
    cmd_pub = rospy.Publisher('/cmd_vel',Twist ,queue_size=1)

    control_pub = rospy.Publisher('llm_control', Int32, queue_size=1)

    cmd_vel = Twist()
    control = Int32()

    rate=rospy.Rate(1) # topic frequency
    while not rospy.is_shutdown():
        try:
            text_input = input("请开始:")
            if text_input == 'q':
               break
        except ValueError:
            print("输入格式有误")
            break

        # 录音
        #start_time = time.time()
        #audio_record(5, 'output.wav')

        # whisper转录
        #audio_file_path = "output.wav"
        #user_prompt = whisper_transcribe(audio_file_path)
        
        #user_prompt = input("请输入内容（输入q退出）: ").strip()

        if user_prompt is None:
            rospy.logwarn("Failed to transcribe audio.")
            break
        else:
            print(user_prompt)
       
        try:
            depth_intrin, img_color, img_depth = get_aligned_images()
            saveFile = "imgSave.png"
            cv2.imwrite(saveFile, img_color)
        except:
            print("No video")
            break

        start_time = time.time()
        completion = dashscope.MultiModalConversation.call(
            model = "qwen-vl-max",
            messages = [
                {"role": "system", "content": "你的身份信息:你是一只四足机器狗,名字叫Lite,你是由杭州云深处科技公司研发的。你擅长与人聊天并执行他们给定的运动或交流任务，你的输入和输出语言为中文,"
                                            "你的任务：与使用者实时交流，你需要判断对话中是否包含需要执行的任务，可以执行的任务有下面几项,"
                                            "可执行的任务：1.导航到一个物体附近。2.向左转。3.向右转。4.向前走。5.向后退。6.跳舞。7.你好 8.表演才艺"
                                            "当与你交流的人需要你执行这些任务时候，你需要给出任务的对应编号，当你判断这个任务编号是1的时候，你还需要根据提供的图片识别是否存在需要导航任务要去的物体，假设这个物体是X，如果存在则输出数字1以及那个物体名字的小写英文，不需要其他的词或者句子，如果不存在，则输出：‘我没看到X’。"
                                            "除此之外的任务请求你需要给出拒绝的回答，如果你判断当前的对话不是需要处理的任务，就正常进行交流。"
                                            "举例：你收到的对话为：导航到箱子附近。如果你收到的图片里有箱子，那么你需要输出：1box，如果你收到的图片里没有箱子，那么你需要输出：我没看到箱子。"
                                            "举例：你收到的对话为：导航到椅子附近。如果你收到的图片里有椅子，那么你需要输出：1chair，如果你收到的图片里没有椅子，那么你需要输出：我没看到椅子。"
                                            "举例：你收到的对话为：导航到人附近。那么你需要输出：1person。"
                                            "举例：你收到的对话为：直行，那么你需要输出数字4."
                                            "举例：你收到的对话为：你可以跳个舞吗，那么你需要输出数字6."
                                            "举例：你收到的对话为：你今天吃饭了吗，你判断这不是一个任务请求，所以你输出正常的交流内容，例如：我还没吃饭."
                                            
                                            
                                            },
                {"role": "user", "content": [
                    {
                        "text":user_prompt
                    },
                    {
                        "image":"imgSave.png"
                    }
                ]}
            ]
        )
        first_txt = completion.output.choices[0].message.content[0]['text'][0]
        print(f"大模型推理时长: {time.time() - start_time:.2f} s")

        if first_txt == '1':
            goal_reached = 1
            print("好的，开始执行")
            object_name = completion.output.choices[0].message.content[0]['text'].strip('1')

            while goal_reached==1:
                depth_intrin, img_color, img_depth = get_aligned_images()
                start_time = time.time()
                results = model.predict(img_color, save=False, show_conf=False)
                #print(f"YOLO detection time: {time.time() - start_time:.2f} seconds")

                for result in results:
                    boxes = result.boxes.xywh.tolist()
                    im_array = result.plot()
                    for i in range(len(boxes)):
                        ux, uy, w, h = boxes[i]
                        class_id = int(result.boxes.cls[i])
                        label = model.names[class_id]
                    
                        if label == object_name:
                            ux, uy = int(ux), int(uy)
                            depth = img_depth[uy, ux]
                    
                            # 获取相机坐标系下的三维坐标
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [ux, uy], depth)
                            camera_xyz = np.round(camera_xyz, 3)  # 保留三位小数
                        

                            # 转换为机器狗的中心坐标
                            robot_x = (depth / 1000.0) - 0.3  # 减去摄像头到机器狗nx中心距离
                            robot_y = (-ux  / 1000.0) + 0.405  # 转换为米并调整坐标
                            robot_z = -uy / 1000.0  # 转换为米并调整坐标

                            print(f"Robot coordinates: [{robot_x}, {robot_y}, {robot_z}]")

                            cmd_vel.linear.x = 0.25 * robot_x
                            cmd_vel.linear.y = 0.1 * robot_y
                            cmd_vel.angular.z = 0
                            cmd_pub.publish(cmd_vel)

                            if robot_x <= 0.3:
                                goal_reached = 0

                            time.sleep(0.25)

        elif first_txt == '2':
            print("好的，开始执行")
            cmd_vel.linear.x = 0
            cmd_vel.linear.y = 0
            cmd_vel.angular.z = 3.0
            ff = 30
            while ff > 0:
                cmd_pub.publish(cmd_vel)
                ff = ff -1 
        elif first_txt == '3':
            print("好的，开始执行")
            cmd_vel.linear.x = 0
            cmd_vel.linear.y = 0
            cmd_vel.angular.z = -3.0
            ff = 30
            while ff > 0:
                cmd_pub.publish(cmd_vel)
                ff = ff -1 
        elif first_txt == '4':
            print("好的，开始执行")
            cmd_vel.linear.x = 0.5
            cmd_vel.linear.y = 0
            cmd_vel.angular.z = 0.0
            ff = 5
            while ff > 0:
                cmd_pub.publish(cmd_vel)
                ff = ff -1 
        elif first_txt == '5':
            print("好的，开始执行")
            cmd_vel.linear.x = -0.5
            cmd_vel.linear.y = 0
            cmd_vel.angular.z = 0.0
            ff = 5
            while ff > 0:
                cmd_pub.publish(cmd_vel)
                ff = ff -1 
        elif first_txt == '6':
            print("好的，我来跳个舞")
            control.data = 6
            control_pub.publish(control)
        elif first_txt == '7':
            print("你好")
            control.data = 7
            control_pub.publish(control)
        elif first_txt == '8':
            print("好的，看我表演")
            control.data = 8
            control_pub.publish(control)
        else:
            outputTxt = completion.output.choices[0].message.content[0]['text']
            print(outputTxt)

        rate.sleep()


