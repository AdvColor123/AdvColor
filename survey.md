-- This is part of the survey questions. Some descriptive contents or image examples for specific adversarial attacks are omitted to avoid redundancy.

+ What is your age?
<br> * 18-25
<br> * 26-35
<br> * 36-45
<br> * > 45


+ What type of company are you in?
<br> * Multinational company
<br> * State-owned company
<br> * Private company
<br> * Other
   
+ What is your position?
<br> * Algorithm/Testing Engineer 
<br> * Project/Product Manager
<br> * (Deputy) General Manager
<br> * Other
       

+ What is your specific job mainly related to?
<br> * Agricultural technology
<br> * Augmented/Virtual reality
<br> * Autonomous driving
<br> * Facial recognition
<br> * Financial technology
<br> * Smart city
<br> * Smart healthcare
<br> * Other

+ What are you currently mainly engaged in? Business/Vocational work or research work?
<br> * Vocational work only
<br> * Mainly focused on vocational work, but also exposed to some academic work
<br> * Research work only
      
+ How familiar are you with adversarial attacks?
<br> * Completely unknown
<br> * Slightly familiar
<br> * Very familiar
      

+ Current adversarial attacks include restricted attacks that are imperceptible to the human eye and unrestricted attacks that are observable to the human eye but might be imperceptible. Unrestricted attacks include attacks based on patch, watermark, optical, style, texture, color, etc. Besides, we propose a perceptible but effective attack called AdvColor. *Detailed method introduction and visualized adversarial examples are omitted here.* Which types of attacks do you think are more harmful in real-world scenarios? You can choose more than one answer.
<br> * Restricted attacks
<br> * Patch-based attacks
<br> * Watermark-based attacks
<br> * Optical-based attacks
<br> * Style-based attacks
<br> * Texture-based attacks
<br> * Color-based attacks (AdvColor excluded)
<br> * AdvColor

+ Which types of attacks do you think are easier to implement in real-world scenarios? For example, it can be achieved by using simple devices in daily life.
<br> * Restricted attacks
<br> * Patch-based attacks
<br> * Watermark-based attacks
<br> * Optical-based attacks
<br> * Style-based attacks
<br> * Texture-based attacks
<br> * Color-based attacks (AdvColor excluded)
<br> * AdvColor

+ From the perspective of perturbation type, rate the perceptibility of each type of attack from 1 to 10. A higher score indicates that the perturbations are more obvious.

+ From the perspective of digital attacks, rate the simplicity of each type of attack in digital implementation (e.g., optimization speed, computational complexity, query budget) from 1 to 5. *The detailed theoretical introduction to each attack is omitted here.* A higher score indicates that the attack is simpler.

+ What are the main factors when considering whether an attack is harmful in real life? You can choose more than one answer.
<br> * The equipment required for the attack is simple.
<br> * Low cost of equipment required for attacks.
<br> * Low threshold for equipment usage and knowledge.
<br> * The technical principles are simple and easy to understand.
<br> * Ability to achieve short-term mass production of adversarial samples.
<br> * Others: \_\_\_\_\_\_


+ We simply summarize the evaluation criteria for the threat of attacks as weak professionalism, ease of mastery, and ease of operation. 

    **Weak professionalism (WP)**: Attackers do not need to master relevant professional knowledge, such as deep learning, image recognition, \etc. The threshold for attack is low, and attacks that ordinary people can consciously or unconsciously launch are more harmful. 
    
    **Ease of mastery (EM)**: Through self-learning or being taught by others, attackers can easily implement and quickly master this form of attack.
    
    **Ease of operation (EO)**: The attack equipment is low-cost and easy to deploy, and adversarial samples can be produced in large quantities in a short period of time.
    
    Based on the degree of threat caused by different attacks in real life, rank the importance of these three factors. '>' indicates that the former is more important than the latter.
<br> * Weak professionalism>Ease of mastery>Ease of operation
<br> * Weak professionalism>Ease of operation>Ease of mastery
<br> * Ease of mastery>Weak professionalism>Ease of operation
<br> * Ease of mastery>Ease of operation>Weak professionalism
<br> * Ease of operation>Weak professionalism>Ease of mastery
<br> * Ease of operation>Ease of mastery>Weak professionalism

+ Rate the three factors from 1 to 5 for the aforementioned attacks. Please pay attention to the correspondence between the evaluation factors and the scores. 
    
    **Weak professionalism**: The lower the requirement for professional knowledge, the higher the score. 
    
    **Easy of mastery**: If attacks are easier to master and the score is higher. 
    
    **Ease of operation**: If attacks are easier to operate, more capable of mass production, and the score is higher. 
    
    For example, if you believe that an attack with the greatest threat needs to meet weak professionalism, easy to master, and easy to operate simultaneously, the corresponding score should be 5, 5, and 5.

+ Some black companies use simple techniques to produce adversarial samples in large quantities. Do you think that the threat to the product in business is significant?
<br> * There is almost no impact.
<br> * There is a certain impact, but it is not currently observed and may become a major threat in the future.
<br> * The impact is significant and some customers have already reported similar issues.
<br> * The impact is extremely significant and it is necessary to develop defense strategies as soon as possible to address this challenge.


+ We propose an idea to achieve color attack, which involves purchasing a light source or smart light bulb and placing it in a corner of a room or near an access control system to continuously emit one or more colors of light so that all images taken by employees passing through the access control system pass through a certain color filter. We found through experiments that publicly available face anti-spoofing/recognition models in academia are sensitive to certain colors and could achieve a large number of misclassifications. This form of attack is simple, has a low threshold, low cost, and is a stable and long-term security threat. May I ask your opinion on this form of attack? Do you think that in business scenarios, we should focus more on attacks that are perceptible but harmful, rather than those attacks which exploit complex mathematical principles to keep imperceptibility and are difficult to implement in real life?
    
+ Do you think we should prioritize defense against perceptible attacks such as AdvColor over defense against restricted/imperceptible attacks? Explain briefly the reason.
    
+ Imagine a scenario where the model used by the customer (with sufficient accuracy in the test set) is found to have security issues (e.g., when shining purple light on all spoofing faces, they will be misclassified as bonafide in most cases). What do you think is the reason for this phenomenon? What is the possible solution? Are you concerned that the business model may be generally sensitive to certain colors?
    
+ From your perspective, what would the consequences or impacts be if perceptible attacks such as AdvColor occur on the products you develop or handle? You can consider aspects such as the business itself, customers, product security, etc. 
