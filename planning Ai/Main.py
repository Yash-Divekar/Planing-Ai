import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = {
    'machine_speed': np.random.normal(50, 5, 1000),
    'temperature': np.random.normal(200, 10, 1000),
    'vibration': np.random.normal(5, 1, 1000),
    'material_usage': np.random.normal(100, 15, 1000),
    'production_output': np.random.normal(500, 50, 1000)
}

df = pd.DataFrame(data)


X = df[['machine_speed', 'temperature', 'vibration', 'material_usage']]
y = df['production_output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


class Generator(nn.Module):
    def _init_(self, input_dim, output_dim):
        super(Generator, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def _init_(self, input_dim):
        super(Discriminator, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


generator = Generator(input_dim=4, output_dim=4)
discriminator = Discriminator(input_dim=4)


g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)


for epoch in range(1000):
    noise = torch.randn(32, 4)
    fake_data = generator(noise)
    
    # Train discriminator
    real_data = torch.tensor(X_train.sample(32).values, dtype=torch.float)
    d_loss = -torch.mean(torch.log(discriminator(real_data)) + torch.log(1 - discriminator(fake_data)))
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Train generator
    g_loss = -torch.mean(torch.log(discriminator(fake_data.detach())))
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()


def get_user_input():
    num_orders = int(input("Enter the number of orders: "))
    orders = []
    for i in range(num_orders):
        print(f"Order {i+1}:")
        size = input("Enter the size of the nut/bolt: ")
        metal_type = input("Enter the metal type: ")
        quantity = int(input("Enter the quantity: "))
        deadline = input("Enter the deadline (YYYY-MM-DD): ")
        profit = float(input("Enter the expected profit: "))
        orders.append({
            'size': size,
            'metal_type': metal_type,
            'quantity': quantity,
            'deadline': deadline,
            'profit': profit
        })
    return orders


def adjust_manufacturing_parameters(orders):
    adjusted_parameters = []
    for order in orders:
        
        speed_adjustment = np.random.uniform(0.9, 1.1)
        temp_adjustment = np.random.uniform(0.95, 1.05)
        vibration_adjustment = np.random.uniform(0.98, 1.02)
        material_usage_adjustment = np.random.uniform(0.95, 1.05)

        adjusted_parameters.append({
            'order': order,
            'adjusted_machine_speed': speed_adjustment * np.mean(df['machine_speed']),
            'adjusted_temperature': temp_adjustment * np.mean(df['temperature']),
            'adjusted_vibration': vibration_adjustment * np.mean(df['vibration']),
            'adjusted_material_usage': material_usage_adjustment * np.mean(df['material_usage'])
        })
    return adjusted_parameters


orders = get_user_input()


adjusted_parameters = adjust_manufacturing_parameters(orders)


for param in adjusted_parameters:
    print(f"\nOrder for {param['order']['quantity']} {param['order']['metal_type']} {param['order']['size']} nuts/bolts:")
    print(f"  Adjusted Machine Speed: {param['adjusted_machine_speed']:.2f}")
    print(f"  Adjusted Temperature: {param['adjusted_temperature']:.2f}")
    print(f"  Adjusted Vibration: {param['adjusted_vibration']:.2f}")
    print(f"  Adjusted Material Usage: {param['adjusted_material_usage']:.2f}")

