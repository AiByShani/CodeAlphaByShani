import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = [
    # General Company Information
    ("What services does your software house provide?", 
     "We offer custom software development, mobile and web application development, cloud solutions, and software consulting services."),
    
    ("What industries do you specialize in?", 
     "We specialize in various industries including healthcare, finance, e-commerce, education, logistics, and more."),
    
    ("How long have you been in the software development industry?", 
     "We have been in the software development industry for over 10 years."),
    
    ("What technologies do you specialize in?", 
     "We specialize in technologies like Python, Java, .NET, React, Angular, Node.js, Django, Flask, AWS, Azure, and more."),
    
    ("Can you provide case studies or success stories?", 
     "Yes, we have several case studies available on our website that highlight our successful projects and satisfied clients."),
    
    # Services Offered
    ("Do you offer mobile app development?", 
     "Yes, we offer both Android and iOS app development services tailored to meet your specific business needs."),
    
    ("Do you provide software maintenance and support?", 
     "Yes, we provide comprehensive software maintenance and support services to ensure your application runs smoothly."),
    
    ("Do you offer cloud solutions?", 
     "Yes, we offer cloud-based solutions including migration, integration, and cloud-native application development."),
    
    ("Can you help with legacy software modernization?", 
     "Yes, we specialize in modernizing legacy systems by migrating them to modern platforms and frameworks."),
    
    # Development Process
    ("What is your software development process?", 
     "Our development process includes requirement analysis, design, development, testing, deployment, and post-launch support."),
    
    ("How do you ensure the quality of the software?", 
     "We follow industry-standard quality assurance practices, including automated testing, code reviews, and continuous integration."),
    
    ("What project management methodologies do you use?", 
     "We use Agile, Scrum, and Waterfall methodologies based on the project requirements and client preferences."),
    
    ("Can I be involved in the development process?", 
     "Yes, we encourage client involvement at every stage of the development process to ensure the final product meets your expectations."),
    
    ("How do you handle changes in project scope?", 
     "We manage changes through a well-defined change control process and communicate any impact on timeline and cost with the client."),
    
    # Pricing and Payment
    ("How do you determine the cost of a project?", 
     "The cost is determined based on the project scope, complexity, technology stack, and required resources."),
    
    ("What are your payment terms?", 
     "We typically require a percentage of the project cost upfront, with the remaining balance payable in milestones or upon completion."),
    
    ("Do you offer flexible pricing models?", 
     "Yes, we offer various pricing models including fixed price, hourly rate, and dedicated team models."),
    
    ("Are there any hidden charges?", 
     "No, we provide transparent pricing and there are no hidden charges. Any additional costs are discussed and approved by the client."),
    
    # Security and Confidentiality
    ("How do you ensure the security of my data?", 
     "We follow strict security protocols, including data encryption, secure access controls, and regular security audits to protect your data."),
    
    ("Do you sign a Non-Disclosure Agreement (NDA)?", 
     "Yes, we are open to signing an NDA to protect your intellectual property and confidential information."),
    
    ("What measures do you take to protect against data breaches?", 
     "We implement security best practices such as firewall protection, intrusion detection systems, and regular vulnerability assessments."),
    
    ("How do you handle data privacy and compliance?", 
     "We comply with data privacy laws such as GDPR and CCPA, and ensure all data handling practices adhere to legal standards."),
    
    # Support and Maintenance
    ("What kind of support do you offer post-deployment?", 
     "We offer ongoing support and maintenance services, including bug fixes, updates, and feature enhancements."),
    
    ("Do you offer 24/7 customer support?", 
     "Yes, we provide 24/7 customer support through phone, chat, and email to address any issues or queries."),
    
    ("What is the response time for support requests?", 
     "We aim to respond to support requests within 24 hours and resolve critical issues as quickly as possible."),
    
    # Miscellaneous
    ("Can you help with digital marketing and SEO?", 
     "Yes, we have a dedicated team that provides digital marketing and SEO services to help you reach a broader audience."),
    
    ("What is the typical timeline for a software development project?", 
     "The timeline depends on the project scope and complexity, but most projects take between 3 to 6 months to complete."),
    
    ("Do you provide training for the software developed?", 
     "Yes, we provide comprehensive training and documentation to help your team use and maintain the software effectively."),
    
    ("How do you handle project cancellations?", 
     "In the event of a project cancellation, we follow our agreement terms regarding refunds and deliverables completed up to that point."),
    
    ("What platforms do you support for mobile app development?", 
     "We support both native (iOS and Android) and cross-platform development (React Native, Flutter, Xamarin).")
]


nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def get_best_match(user_questeion, faqs):
    processed_faqs = [preprocess(faq[0]) for faq in faqs]
    processed_user_question = preprocess(user_questeion)

    # Vectorize the Text
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_faqs+[processed_user_question])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])


    # Match faq with higest similarity
    best_match_idx = cosine_sim.argmax()
    return faqs[best_match_idx][1], cosine_sim[0,best_match_idx]


def chatbot():
    print("Welcome to the FAQ chatbot! Ask me anything.")
    while True:
        user_question = input("You:")
        if user_question.lower() in ["exit","quit"]:
            print("Goodbye!")
            break

        answer, confidence = get_best_match(user_question,faqs)
        if confidence > 0.5:
            print(f"Bot: {answer}")
        else:
            print("Bot: I'm not sure about that. Could you ask something else?")


if __name__ == "__main__":
    chatbot()


